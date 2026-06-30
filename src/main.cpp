// volsmoke — 3D ボリュメトリック煙＆炎の GPU シミュレーション + レイマーチ描画。
//
// 自作エンジン MitiruEngine の DX12 compute 基盤 (Dx12ComputeContext) を土台に、
// 3D グリッド上で Navier-Stokes (Stam の安定流体) を解き、密度・温度場を
// レイマーチで立体描画する。
//
//   注入 → 移流 → 浮力 → 渦confinement → 発散 → 圧力Jacobi → 射影 → レイマーチ
//
// 1 つの sim は Field 構造体が全状態 (テクスチャ/CBV/バインディング) を持つ。
// 複数 Field を横に並べて描く compare モードで「いろんな炎/煙」を一画面で比較できる。

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <mitiru/render/dx12/Dx12ComputeContext.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dwmapi.lib")

using Microsoft::WRL::ComPtr;
using mitiru::render::dx12::Dx12ComputeContext;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
static bool g_quit = false;

// ImGui が要求する SRV ディスクリプタを小プールから貸し出す (1.92 は複数要求しうる)
struct ImguiSrvPool
{
    ID3D12DescriptorHeap* heap = nullptr;
    UINT inc = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu0{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu0{};
    std::vector<UINT> freeIdx;
} g_srv;

static void imguiSrvAlloc(ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* outCpu, D3D12_GPU_DESCRIPTOR_HANDLE* outGpu)
{
    UINT i = g_srv.freeIdx.back(); g_srv.freeIdx.pop_back();
    outCpu->ptr = g_srv.cpu0.ptr + SIZE_T(i) * g_srv.inc;
    outGpu->ptr = g_srv.gpu0.ptr + UINT64(i) * g_srv.inc;
}
static void imguiSrvFree(ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE)
{
    UINT i = UINT((cpu.ptr - g_srv.cpu0.ptr) / g_srv.inc);
    g_srv.freeIdx.push_back(i);
}

static LRESULT CALLBACK volWndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    if (ImGui_ImplWin32_WndProcHandler(h, m, w, l)) return 1;
    if (m == WM_CLOSE || m == WM_DESTROY) { g_quit = true; return 0; }
    return DefWindowProcW(h, m, w, l);
}

static void check(HRESULT hr, const char* what)
{
    if (FAILED(hr))
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s failed: hr=0x%08lx", what,
                      static_cast<unsigned long>(hr));
        throw std::runtime_error(buf);
    }
}

// ── GPU 定数バッファ (HLSL の cbuffer P と byte 一致) ─────────────────────
struct alignas(16) GpuParams
{
    std::uint32_t gridN;  float time;   float dt;      float frameRnd;   // row0
    float camPos[3];      float fovTan;                                  // row1
    float camTar[3];      float aspect;                                  // row2
    std::int32_t imgW;    std::int32_t imgH; float buoyancy; float cooling; // row3
    float densDissip;     float velDissip; float emitRadius; float emitTemp; // row4
    float emitDensity;    float emitPos[3];                              // row5
    float wind[3];        float emitVel;                                 // row6
    float vorticity;      float light[3];                                // row7
    float smokeColor[3];  float fireGain;                                // row8
    float tempScale;      float fireMode; float ambient; float absorb;   // row9
    float pad[24];                                                       // → 256B
};
static_assert(sizeof(GpuParams) == 256, "GpuParams must be 256 bytes");

// HLSL に前置する共通 cbuffer 宣言 (全シェーダで同一)
static const char* kCB = R"HLSL(
cbuffer P : register(b0) {
    uint gridN; float time; float dt; float frameRnd;
    float3 camPos; float fovTan; float3 camTar; float aspect;
    int imgW; int imgH; float buoyancy; float cooling;
    float densDissip; float velDissip; float emitRadius; float emitTemp;
    float emitDensity; float3 emitPos; float3 wind; float emitVel;
    float vorticity; float3 light;
    float3 smokeColor; float fireGain;
    float tempScale; float fireMode; float ambient; float absorb;
};
int3 clampi(int3 c){ return clamp(c, int3(0,0,0), int3(gridN-1,gridN-1,gridN-1)); }
#define TRILERP(TEX,P,OUT) { float3 _q=clamp((P),0.0,(float)gridN-1.001); int3 _b=int3(floor(_q)); float3 _f=frac(_q); OUT=0; [unroll] for(int _z=0;_z<2;++_z) [unroll] for(int _y=0;_y<2;++_y) [unroll] for(int _x=0;_x<2;++_x){ int3 _c=clampi(_b+int3(_x,_y,_z)); float _w=(_x?_f.x:1.0-_f.x)*(_y?_f.y:1.0-_f.y)*(_z?_f.z:1.0-_f.z); OUT+=TEX[_c]*_w; } }
)HLSL";

// ── 1 つの流体シム = テクスチャ群 + CBV + バインディング + パラメータ ──────
struct Field
{
    template <typename T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    ComPtr<ID3D12Resource> velA, velB, denA, denB, tmpA, tmpB, prsP, prsQ, div, curl, output;
    ComPtr<ID3D12Resource> cb; std::uint8_t* cbPtr = nullptr;
    GpuParams params{};
    std::string preset = "fire";
    bool inited = false;
    int outW = 0, outH = 0;
    D3D12_GPU_DESCRIPTOR_HANDLE setAdvectVel{}, setAdvectScalar{}, setInject{}, setBuoyancy{},
        setCurl{}, setVort{}, setDivergence{}, setClearP{}, setJacobi0{}, setJacobi1{},
        setProject{}, setCopyVel{}, setCopyDen{}, setCopyTmp{}, setRaymarch{};
};

// ── アプリ本体 ───────────────────────────────────────────────────────────
class VolSmoke
{
public:
    // ヘッドレス capture (presets が複数なら横並び composite を焼く)
    void run(const std::string& outPath, int frames, int gridN, int w, int h,
             const std::vector<std::string>& presets,
             const std::string& seqDir = "", int seqFrom = 0, int seqStride = 1)
    {
        m_gridN = gridN; const int n = (int)presets.size(); setSize(w, h, n);
        createDevice(); createCmd(); createHeap(); buildShaders();
        createComposite();
        m_sims.resize(n);
        for (int i = 0; i < n; ++i) { m_sims[i].preset = presets[i]; createField(m_sims[i], m_tileW, m_imgH); }

        const bool dump = !seqDir.empty();
        for (int f = 0; f < frames; ++f)
        {
            const bool render = dump || (f == frames - 1);
            for (auto& fld : m_sims) updateField(fld, f);
            recordFrame(render);
            submitAndWait();
            if (dump && f >= seqFrom && ((f - seqFrom) % seqStride == 0))
            {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/f_%04d.png", seqDir.c_str(), f);
                readbackAndSave(path);
            }
        }
        if (!dump) readbackAndSave(outPath);
        std::printf("[volsmoke] wrote %s (%dx%d, grid=%d^3, %d frames)\n",
                    outPath.c_str(), m_imgW, m_imgH, m_gridN, frames);
    }

    // 窓モード: スライダーでライブ調整 / compare で複数を横並び。
    // selftest = 窓を表示せず数十フレーム回して終了 (経路のクラッシュ検証用)。
    void runInteractive(int gridN, const std::vector<std::string>& presets,
                        int w, int h, bool selftest = false)
    {
        m_gridN = gridN; m_selftest = selftest;
        const int n = (int)presets.size();
        setSize(w, h, n);
        createDevice(); createCmd(); createHeap(); buildShaders();
        createComposite();
        m_sims.resize(n);
        for (int i = 0; i < n; ++i)
        {
            m_sims[i].preset = presets[i];
            createField(m_sims[i], m_tileW, m_imgH);
        }
        createWindowAndSwap();
        initImGui();

        const int maxFrames = selftest ? 45 : 0;
        int frame = 0;
        while (!g_quit)
        {
            pumpMessages();
            if (g_quit) break;
            if (maxFrames && frame >= maxFrames) break;

            ImGui_ImplDX12_NewFrame(); ImGui_ImplWin32_NewFrame(); ImGui::NewFrame();
            drawUI();
            ImGui::Render();

            // selftest は rebuild 経路も踏んでおく
            if (selftest && frame == 20) m_pending = {"fire","smoke"};
            if (!m_pending.empty()) { rebuildSims(m_pending); m_pending.clear(); }

            for (auto& f : m_sims) updateField(f, frame);
            recordFrame(true);
            presentComposite();
            ++frame;
        }
        waitForGpu();
        ImGui_ImplDX12_Shutdown(); ImGui_ImplWin32_Shutdown(); ImGui::DestroyContext();
        if (selftest) std::printf("[volsmoke] interactive selftest OK (%d frames, %d sims, no window)\n",
                                  frame, (int)m_sims.size());
    }

private:
    static constexpr DXGI_FORMAT kScalarFmt = DXGI_FORMAT_R16_FLOAT;
    static constexpr DXGI_FORMAT kVecFmt    = DXGI_FORMAT_R16G16B16A16_FLOAT;
    static constexpr DXGI_FORMAT kPrsFmt    = DXGI_FORMAT_R32_FLOAT;
    static constexpr DXGI_FORMAT kImgFmt    = DXGI_FORMAT_R8G8B8A8_UNORM;
    static constexpr int kJacobi = 40;
    static constexpr UINT kHeapSize = 768;   // sim 数 × 15 set × 8

    void setSize(int w, int h, int nSims)
    {
        m_tileW = w / nSims;
        m_imgW  = m_tileW * nSims;   // タイル整数割りに合わせる
        m_imgH  = h;
    }

    // 窓を保ったまま sim 構成を作り直す (UI からのレイアウト/プリセット切替)。
    // 窓・swapchain・composite (m_imgW×m_imgH) は不変、Field 群だけ再生成。
    void rebuildSims(const std::vector<std::string>& presets)
    {
        waitForGpu();                 // 旧リソースが GPU で使用中でないことを保証
        m_sims.clear();
        m_cursor = 0;                 // descriptor heap を先頭から再利用
        const int n = presets.empty() ? 1 : (int)presets.size();
        m_tileW = m_imgW / n;
        m_sims.resize(n);
        for (int i = 0; i < n; ++i) { m_sims[i].preset = presets[i]; createField(m_sims[i], m_tileW, m_imgH); }
    }

    // ── デバイス / コマンド / ヒープ ───────────────────────────────────
    void createDevice()
    {
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)),
              "D3D12CreateDevice");
        D3D12_FEATURE_DATA_D3D12_OPTIONS opt{};
        m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opt, sizeof(opt));
        if (!opt.TypedUAVLoadAdditionalFormats)
            std::fprintf(stderr, "[volsmoke] WARN: TypedUAVLoadAdditionalFormats=FALSE\n");
        D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)), "CreateCommandQueue");
    }
    void createCmd()
    {
        check(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_alloc)),
              "CreateCommandAllocator");
        check(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_alloc.Get(), nullptr,
                                          IID_PPV_ARGS(&m_cl)), "CreateCommandList");
        m_cl->Close();
        check(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)), "CreateFence");
        m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }
    void createHeap()
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = kHeapSize; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_heap)), "CreateDescriptorHeap");
        m_inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT i) const
    { auto h = m_heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i)*m_inc; return h; }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const
    { auto h = m_heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i)*m_inc; return h; }

    // ── リソース生成 ──────────────────────────────────────────────────
    ComPtr<ID3D12Resource> makeTex3D(DXGI_FORMAT fmt)
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        d.Width = UINT64(m_gridN); d.Height = UINT(m_gridN);
        d.DepthOrArraySize = UINT16(m_gridN); d.MipLevels = 1;
        d.Format = fmt; d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> r;
        check(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
              D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&r)), "tex3d");
        return r;
    }
    ComPtr<ID3D12Resource> makeTex2D(int w, int h, D3D12_RESOURCE_STATES st)
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = UINT64(w); d.Height = UINT(h); d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.Format = kImgFmt; d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> r;
        check(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr,
              IID_PPV_ARGS(&r)), "tex2d");
        return r;
    }
    void uav3D(ID3D12Resource* res, DXGI_FORMAT fmt, UINT slot)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = fmt; d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        d.Texture3D.WSize = UINT(m_gridN);
        m_device->CreateUnorderedAccessView(res, nullptr, &d, cpu(slot));
    }
    void uav2D(ID3D12Resource* res, UINT slot)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = kImgFmt; d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m_device->CreateUnorderedAccessView(res, nullptr, &d, cpu(slot));
    }

    ComPtr<ID3D12Resource> makeCB()
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sizeof(GpuParams); d.Height = 1;
        d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> r;
        check(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
              D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&r)), "cb");
        return r;
    }

    void createComposite()
    {
        m_composite = makeTex2D(m_imgW, m_imgH, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }

    struct Tex { ID3D12Resource* res; DXGI_FORMAT fmt; bool is3D; };
    D3D12_GPU_DESCRIPTOR_HANDLE makeUavSet(std::initializer_list<Tex> uavs)
    {
        const UINT base = m_cursor; m_cursor += 8;
        const Tex first = *uavs.begin(); UINT i = 0;
        for (UINT s = 0; s < 8; ++s)
        {
            const Tex t = (i < uavs.size()) ? *(uavs.begin()+i) : first;
            if (t.is3D) uav3D(t.res, t.fmt, base+s); else uav2D(t.res, base+s);
            ++i;
        }
        return gpu(base);
    }

    // 1 つの Field のテクスチャ・CBV・バインディング領域を作る
    void createField(Field& f, int outW, int outH)
    {
        f.outW = outW; f.outH = outH;
        f.velA = makeTex3D(kVecFmt); f.velB = makeTex3D(kVecFmt);
        f.denA = makeTex3D(kScalarFmt); f.denB = makeTex3D(kScalarFmt);
        f.tmpA = makeTex3D(kScalarFmt); f.tmpB = makeTex3D(kScalarFmt);
        f.prsP = makeTex3D(kPrsFmt); f.prsQ = makeTex3D(kPrsFmt);
        f.div  = makeTex3D(kPrsFmt); f.curl = makeTex3D(kVecFmt);
        f.output = makeTex2D(outW, outH, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        f.cb = makeCB();
        D3D12_RANGE none{0,0}; f.cb->Map(0, &none, reinterpret_cast<void**>(&f.cbPtr));

        f.setAdvectVel    = makeUavSet({ {f.velB.Get(),kVecFmt,true}, {f.velA.Get(),kVecFmt,true} });
        f.setAdvectScalar = makeUavSet({ {f.denB.Get(),kScalarFmt,true}, {f.tmpB.Get(),kScalarFmt,true},
                                         {f.denA.Get(),kScalarFmt,true}, {f.tmpA.Get(),kScalarFmt,true},
                                         {f.velA.Get(),kVecFmt,true} });
        f.setInject       = makeUavSet({ {f.denB.Get(),kScalarFmt,true}, {f.tmpB.Get(),kScalarFmt,true}, {f.velB.Get(),kVecFmt,true} });
        f.setBuoyancy     = makeUavSet({ {f.velB.Get(),kVecFmt,true}, {f.tmpB.Get(),kScalarFmt,true} });
        f.setCurl         = makeUavSet({ {f.curl.Get(),kVecFmt,true}, {f.velB.Get(),kVecFmt,true} });
        f.setVort         = makeUavSet({ {f.velB.Get(),kVecFmt,true}, {f.curl.Get(),kVecFmt,true} });
        f.setDivergence   = makeUavSet({ {f.div.Get(),kPrsFmt,true}, {f.velB.Get(),kVecFmt,true} });
        f.setClearP       = makeUavSet({ {f.prsP.Get(),kPrsFmt,true} });
        f.setJacobi0      = makeUavSet({ {f.prsQ.Get(),kPrsFmt,true}, {f.prsP.Get(),kPrsFmt,true}, {f.div.Get(),kPrsFmt,true} });
        f.setJacobi1      = makeUavSet({ {f.prsP.Get(),kPrsFmt,true}, {f.prsQ.Get(),kPrsFmt,true}, {f.div.Get(),kPrsFmt,true} });
        f.setProject      = makeUavSet({ {f.velB.Get(),kVecFmt,true}, {f.prsP.Get(),kPrsFmt,true} });
        f.setCopyVel      = makeUavSet({ {f.velA.Get(),kVecFmt,true}, {f.velB.Get(),kVecFmt,true} });
        f.setCopyDen      = makeUavSet({ {f.denA.Get(),kScalarFmt,true}, {f.denB.Get(),kScalarFmt,true} });
        f.setCopyTmp      = makeUavSet({ {f.tmpA.Get(),kScalarFmt,true}, {f.tmpB.Get(),kScalarFmt,true} });
        f.setRaymarch     = makeUavSet({ {f.output.Get(),kImgFmt,false}, {f.denB.Get(),kScalarFmt,true}, {f.tmpB.Get(),kScalarFmt,true} });
    }

    // ── パラメータ ────────────────────────────────────────────────────
    void setCommon(GpuParams& p, int outW, int outH) const
    {
        p.gridN = std::uint32_t(m_gridN);
        p.camPos[0]=1.45f; p.camPos[1]=0.78f; p.camPos[2]=1.7f;
        p.fovTan = 0.34f;
        p.camTar[0]=0.5f; p.camTar[1]=0.55f; p.camTar[2]=0.5f;
        p.aspect = float(outW)/outH; p.imgW=outW; p.imgH=outH;
        p.emitPos[0]=0.5f; p.emitPos[1]=0.11f; p.emitPos[2]=0.5f;
        p.wind[0]=0; p.wind[1]=0; p.wind[2]=0;
        p.light[0]=0.5f; p.light[1]=0.78f; p.light[2]=0.38f;
        p.velDissip = 0.04f;
    }

    static void applyPreset(GpuParams& p, const std::string& name)
    {
        p.buoyancy=95; p.cooling=1.05f; p.densDissip=0.20f; p.emitRadius=0.13f;
        p.emitTemp=8.0f; p.emitDensity=2.2f; p.emitVel=28; p.vorticity=6.0f;
        p.smokeColor[0]=0.55f; p.smokeColor[1]=0.57f; p.smokeColor[2]=0.62f;
        p.fireGain=60; p.tempScale=0.25f; p.fireMode=0; p.ambient=0.22f; p.absorb=40;

        if (name=="smoke") {
            p.buoyancy=52; p.cooling=0.30f; p.densDissip=0.05f; p.emitRadius=0.14f;
            p.emitTemp=1.4f; p.emitDensity=5.0f; p.emitVel=20; p.vorticity=5.0f;
            p.fireGain=0; p.ambient=0.34f; p.absorb=46;
            p.smokeColor[0]=0.62f; p.smokeColor[1]=0.64f; p.smokeColor[2]=0.70f;
        } else if (name=="blue") {
            p.buoyancy=120; p.cooling=1.7f; p.densDissip=0.4f; p.emitRadius=0.10f;
            p.emitTemp=9.0f; p.emitDensity=1.0f; p.emitVel=34; p.vorticity=7.0f;
            p.fireGain=52; p.tempScale=0.22f; p.fireMode=1; p.absorb=34;
        } else if (name=="ink") {
            p.buoyancy=46; p.cooling=0.3f; p.densDissip=0.04f; p.emitRadius=0.12f;
            p.emitTemp=1.2f; p.emitDensity=5.5f; p.emitVel=18; p.vorticity=6.0f;
            p.fireGain=0; p.ambient=0.30f; p.absorb=48;
            p.smokeColor[0]=0.10f; p.smokeColor[1]=0.55f; p.smokeColor[2]=0.85f;
        } else if (name=="torch") {
            p.buoyancy=150; p.cooling=1.1f; p.densDissip=0.3f; p.emitRadius=0.07f;
            p.emitTemp=9.0f; p.emitDensity=1.6f; p.emitVel=55; p.vorticity=9.0f;
            p.fireGain=70; p.tempScale=0.26f; p.absorb=38;
        }
    }

    // 初回はプリセットから初期化、以降は params を維持 (UI が上書きする)
    void updateField(Field& f, int frame)
    {
        if (!f.inited)
        {
            setCommon(f.params, f.outW, f.outH);
            applyPreset(f.params, f.preset);
            f.inited = true;
        }
        f.params.time = frame/60.0f; f.params.dt = 1.0f/60.0f; f.params.frameRnd = frame*0.6180339887f;
        std::memcpy(f.cbPtr, &f.params, sizeof(GpuParams));
    }

    // ── コマンド記録 ──────────────────────────────────────────────────
    void uavBarrier(ID3D12Resource* r)
    { D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource=r; m_cl->ResourceBarrier(1,&b); }
    void transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
    { D3D12_RESOURCE_BARRIER x{}; x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      x.Transition.pResource=r; x.Transition.StateBefore=a; x.Transition.StateAfter=b;
      x.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; m_cl->ResourceBarrier(1,&x); }

    void disp(Dx12ComputeContext& p, Field& f, D3D12_GPU_DESCRIPTOR_HANDLE set)
    {
        const UINT g = UINT((m_gridN+7)/8);
        p.setCommandList(m_cl.Get());
        p.setRootCBV(0, f.cb->GetGPUVirtualAddress());
        p.setRootUAVTable(set);
        p.dispatch(g, g, g);
    }

    // sim 1 ステップ + (render なら raymarch→f.output)。cl は reset 済 + heap 設定済が前提
    void recordSim(Field& f, bool render)
    {
        disp(*m_advectVel, f, f.setAdvectVel);   uavBarrier(f.velB.Get());
        disp(*m_advectScalar, f, f.setAdvectScalar); uavBarrier(f.denB.Get()); uavBarrier(f.tmpB.Get());
        disp(*m_inject, f, f.setInject);          uavBarrier(f.denB.Get()); uavBarrier(f.tmpB.Get()); uavBarrier(f.velB.Get());
        disp(*m_buoyancy, f, f.setBuoyancy);      uavBarrier(f.velB.Get());
        disp(*m_curlPass, f, f.setCurl);          uavBarrier(f.curl.Get());
        disp(*m_vorticity, f, f.setVort);         uavBarrier(f.velB.Get());
        disp(*m_divergence, f, f.setDivergence);  uavBarrier(f.div.Get());
        disp(*m_clear, f, f.setClearP);           uavBarrier(f.prsP.Get());
        for (int i = 0; i < kJacobi; ++i)
        {
            if ((i & 1) == 0) { disp(*m_jacobi, f, f.setJacobi0); uavBarrier(f.prsQ.Get()); }
            else              { disp(*m_jacobi, f, f.setJacobi1); uavBarrier(f.prsP.Get()); }
        }
        disp(*m_project, f, f.setProject);        uavBarrier(f.velB.Get());
        disp(*m_copyVec, f, f.setCopyVel);        uavBarrier(f.velA.Get());
        disp(*m_copyScalar, f, f.setCopyDen);     uavBarrier(f.denA.Get());
        disp(*m_copyScalar, f, f.setCopyTmp);     uavBarrier(f.tmpA.Get());
        if (render)
        {
            const UINT tx=(f.outW+7)/8, ty=(f.outH+7)/8;
            m_raymarch->setCommandList(m_cl.Get());
            m_raymarch->setRootCBV(0, f.cb->GetGPUVirtualAddress());
            m_raymarch->setRootUAVTable(f.setRaymarch);
            m_raymarch->dispatch(tx, ty, 1);
            uavBarrier(f.output.Get());
        }
    }

    // 全 Field を sim + (render なら) 各 output を composite タイルへ集める
    void recordFrame(bool render)
    {
        m_alloc->Reset(); m_cl->Reset(m_alloc.Get(), nullptr);
        ID3D12DescriptorHeap* heaps[] = {m_heap.Get()};
        m_cl->SetDescriptorHeaps(1, heaps);
        for (auto& f : m_sims) recordSim(f, render);
        if (render)
        {
            transition(m_composite.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            for (int i = 0; i < (int)m_sims.size(); ++i)
            {
                Field& f = m_sims[i];
                transition(f.output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION dl{}; dl.pResource=m_composite.Get();
                dl.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dl.SubresourceIndex=0;
                D3D12_TEXTURE_COPY_LOCATION sl{}; sl.pResource=f.output.Get();
                sl.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; sl.SubresourceIndex=0;
                m_cl->CopyTextureRegion(&dl, UINT(i*m_tileW), 0, 0, &sl, nullptr);
                transition(f.output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            transition(m_composite.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        }
        m_cl->Close();
    }

    void waitForGpu()
    {
        const UINT64 v = ++m_fenceVal;
        m_queue->Signal(m_fence.Get(), v);
        if (m_fence->GetCompletedValue() < v)
        { m_fence->SetEventOnCompletion(v, m_fenceEvent); WaitForSingleObject(m_fenceEvent, INFINITE); }
    }
    void submitAndWait()
    { ID3D12CommandList* l[] = {m_cl.Get()}; m_queue->ExecuteCommandLists(1, l); waitForGpu(); }

    // ── ヘッドレス: composite を PNG 保存 ──────────────────────────────
    void readbackAndSave(const std::string& outPath)
    {
        const UINT rowBytes = m_imgW*4;
        const UINT rowPitch = (rowBytes+255)&~255u;
        const UINT64 total = UINT64(rowPitch)*m_imgH;
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC d{};
        d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=total; d.Height=1;
        d.DepthOrArraySize=1; d.MipLevels=1; d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> rb;
        check(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
              D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb)), "readback");
        m_alloc->Reset(); m_cl->Reset(m_alloc.Get(), nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource=rb.Get();
        dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Footprint.Format=kImgFmt; dst.PlacedFootprint.Footprint.Width=m_imgW;
        dst.PlacedFootprint.Footprint.Height=m_imgH; dst.PlacedFootprint.Footprint.Depth=1;
        dst.PlacedFootprint.Footprint.RowPitch=rowPitch;
        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource=m_composite.Get();
        src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex=0;
        // m_composite は recordFrame 後 COPY_SOURCE のまま → そのまま copy 元に使える
        m_cl->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        m_cl->Close(); submitAndWait();

        std::uint8_t* mapped=nullptr; D3D12_RANGE rr{0,SIZE_T(total)};
        rb->Map(0,&rr,reinterpret_cast<void**>(&mapped));
        std::vector<std::uint8_t> tight(size_t(m_imgW)*m_imgH*4);
        for (int y=0;y<m_imgH;++y)
            std::memcpy(&tight[size_t(y)*rowBytes], mapped+size_t(y)*rowPitch, rowBytes);
        D3D12_RANGE wn{0,0}; rb->Unmap(0,&wn);
        if (!stbi_write_png(outPath.c_str(), m_imgW, m_imgH, 4, tight.data(), m_imgW*4))
            throw std::runtime_error("stbi_write_png failed");
    }

    // ── 窓 + swapchain ────────────────────────────────────────────────
    void createWindowAndSwap()
    {
        WNDCLASSEXW wc{}; wc.cbSize=sizeof(wc); wc.lpfnWndProc=volWndProc;
        wc.hInstance=GetModuleHandleW(nullptr); wc.lpszClassName=L"VolSmokeWnd";
        wc.hCursor=LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
        RegisterClassExW(&wc);
        RECT r{0,0,m_imgW,m_imgH};
        DWORD style = WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX;
        AdjustWindowRect(&r, style, FALSE);
        m_hwnd = CreateWindowW(L"VolSmokeWnd", L"volsmoke - 3D volumetric fire & smoke (DX12 compute)",
            style, CW_USEDEFAULT, CW_USEDEFAULT, r.right-r.left, r.bottom-r.top,
            nullptr, nullptr, wc.hInstance, nullptr);
        if (!m_hwnd) throw std::runtime_error("CreateWindow failed");

        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1(swap)");
        DXGI_SWAP_CHAIN_DESC1 sd{};
        sd.Width=m_imgW; sd.Height=m_imgH; sd.Format=kImgFmt; sd.SampleDesc.Count=1;
        sd.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.BufferCount=3;
        sd.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
        ComPtr<IDXGISwapChain1> sc1;
        check(factory->CreateSwapChainForHwnd(m_queue.Get(), m_hwnd, &sd, nullptr, nullptr, &sc1),
              "CreateSwapChainForHwnd");
        check(sc1.As(&m_swap), "SwapChain As IDXGISwapChain3");
        factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER);

        D3D12_DESCRIPTOR_HEAP_DESC rh{}; rh.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; rh.NumDescriptors=3;
        check(m_device->CreateDescriptorHeap(&rh, IID_PPV_ARGS(&m_rtvHeap)), "rtvHeap");
        m_rtvInc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        for (UINT i=0;i<3;++i)
        {
            check(m_swap->GetBuffer(i, IID_PPV_ARGS(&m_back[i])), "GetBuffer");
            auto h = m_rtvHeap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i)*m_rtvInc;
            m_device->CreateRenderTargetView(m_back[i].Get(), nullptr, h);
        }
        if (!m_selftest) { ShowWindow(m_hwnd, SW_SHOW); UpdateWindow(m_hwnd); }
    }

    void initImGui()
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors=64; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_imguiHeap)), "imguiHeap");
        g_srv.heap=m_imguiHeap.Get();
        g_srv.inc=m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        g_srv.cpu0=m_imguiHeap->GetCPUDescriptorHandleForHeapStart();
        g_srv.gpu0=m_imguiHeap->GetGPUDescriptorHandleForHeapStart();
        g_srv.freeIdx.clear(); for (int i=63;i>=0;--i) g_srv.freeIdx.push_back(UINT(i));

        IMGUI_CHECKVERSION(); ImGui::CreateContext();
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();
        ImGui_ImplWin32_Init(m_hwnd);
        ImGui_ImplDX12_InitInfo ii{};
        ii.Device=m_device.Get(); ii.CommandQueue=m_queue.Get(); ii.NumFramesInFlight=3;
        ii.RTVFormat=kImgFmt; ii.SrvDescriptorHeap=m_imguiHeap.Get();
        ii.SrvDescriptorAllocFn=imguiSrvAlloc; ii.SrvDescriptorFreeFn=imguiSrvFree;
        if (!ImGui_ImplDX12_Init(&ii)) throw std::runtime_error("ImGui_ImplDX12_Init failed");
    }

    void pumpMessages()
    {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        { TranslateMessage(&msg); DispatchMessageW(&msg); if (msg.message==WM_QUIT) g_quit=true; }
    }

    void drawUI()
    {
        static const char* PR[] = {"fire","blue","torch","smoke","ink"};
        const int NPR = 5;
        auto idxOf = [&](const std::string& s){ for(int i=0;i<NPR;++i) if(s==PR[i]) return i; return 0; };
        auto curList = [&](){ std::vector<std::string> v; for(auto& f:m_sims) v.push_back(f.preset); return v; };
        const int n = (int)m_sims.size();

        ImGui::SetNextWindowPos(ImVec2(12,12), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(330, n==1?540.0f:200.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("volsmoke");

        // レイアウト: タイル数を 1〜4 で切替 (シムを作り直す)
        ImGui::TextUnformatted("Layout:"); ImGui::SameLine();
        for (int k=1;k<=4;++k){
            char b[8]; std::snprintf(b,sizeof(b),"%d##L%d",k,k);
            const bool active = (n==k);
            if (active) ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(70,110,200,255));
            if (ImGui::Button(b)){
                const char* fill[4]={"fire","blue","smoke","ink"};
                auto v=curList(); v.resize(k);
                for(int i=n;i<k;++i) v[i]=fill[i%4];
                m_pending=v;
            }
            if (active) ImGui::PopStyleColor();
            ImGui::SameLine();
        }
        ImGui::NewLine(); ImGui::Separator();

        // 各タイルのプリセット選択
        for (int i=0;i<n;++i){
            int cur=idxOf(m_sims[i].preset);
            char lbl[24]; std::snprintf(lbl,sizeof(lbl),"tile %d##t%d",i,i);
            if (ImGui::Combo(lbl,&cur,PR,NPR)){
                if (n==1){ GpuParams& p=m_sims[0].params; keep(p); applyPreset(p,PR[cur]); restore(p);
                           m_blueMode=(p.fireMode>0.5f); m_sims[0].preset=PR[cur]; }
                else { auto v=curList(); v[i]=PR[cur]; m_pending=v; }
            }
        }

        // 単発はフルスライダーでライブ調整
        if (n==1){
            GpuParams& p=m_sims[0].params;
            ImGui::Separator();
            ImGui::SliderFloat("buoyancy",    &p.buoyancy,   0.0f, 220.0f);
            ImGui::SliderFloat("cooling",     &p.cooling,    0.0f, 4.0f);
            ImGui::SliderFloat("vorticity",   &p.vorticity,  0.0f, 16.0f);
            ImGui::SliderFloat("smoke decay", &p.densDissip, 0.0f, 1.0f);
            ImGui::SliderFloat("emit temp",   &p.emitTemp,   0.0f, 12.0f);
            ImGui::SliderFloat("emit density",&p.emitDensity,0.0f, 8.0f);
            ImGui::SliderFloat("emit speed",  &p.emitVel,    0.0f, 90.0f);
            ImGui::SliderFloat("emit radius", &p.emitRadius, 0.03f,0.25f);
            ImGui::SliderFloat("fire glow",   &p.fireGain,   0.0f, 100.0f);
            ImGui::SliderFloat("wind X",      &p.wind[0],  -40.0f, 40.0f);
            ImGui::SliderFloat("wind Z",      &p.wind[2],  -40.0f, 40.0f);
            ImGui::Checkbox("blue flame mode", &m_blueMode); p.fireMode = m_blueMode?1.0f:0.0f;
            ImGui::ColorEdit3("smoke color", p.smokeColor);
        }
        ImGui::Separator();
        ImGui::Text("%.1f FPS  -  %d x 128^3  -  DX12 compute", ImGui::GetIO().Framerate, n);
        ImGui::End();

        // compare 時はタイル上にプリセット名ラベル
        if (n>1){
            auto* dl = ImGui::GetForegroundDrawList();
            for (int i=0;i<n;++i){
                const char* lab = m_sims[i].preset.c_str();
                float x = i*float(m_tileW) + m_tileW*0.5f - ImGui::CalcTextSize(lab).x*0.5f;
                dl->AddText(ImVec2(x, 10), IM_COL32(235,238,242,255), lab);
            }
        }
    }

    // プリセット切替で camera/emit/light/出力情報を壊さない退避・復元
    float m_kc[3],m_kt[3],m_ke[3],m_kl[3],m_kfov=0,m_kasp=0; int m_kw=0,m_kh=0;
    bool  m_blueMode=false;
    std::vector<std::string> m_pending;   // UI からの再構成要求 (次フレーム頭で適用)
    void keep(GpuParams& p){ for(int i=0;i<3;++i){m_kc[i]=p.camPos[i];m_kt[i]=p.camTar[i];m_ke[i]=p.emitPos[i];m_kl[i]=p.light[i];}
        m_kfov=p.fovTan;m_kasp=p.aspect;m_kw=p.imgW;m_kh=p.imgH; }
    void restore(GpuParams& p){ for(int i=0;i<3;++i){p.camPos[i]=m_kc[i];p.camTar[i]=m_kt[i];p.emitPos[i]=m_ke[i];p.light[i]=m_kl[i];}
        p.fovTan=m_kfov;p.aspect=m_kasp;p.imgW=m_kw;p.imgH=m_kh;p.gridN=std::uint32_t(m_gridN);p.velDissip=0.04f; }

    void presentComposite()
    {
        const UINT bb = m_swap->GetCurrentBackBufferIndex();
        m_alloc->Reset(); m_cl->Reset(m_alloc.Get(), nullptr);
        ID3D12DescriptorHeap* ch[] = {m_heap.Get()};
        m_cl->SetDescriptorHeaps(1, ch);
        for (auto& f : m_sims) recordSim(f, true);
        // composite を組む (recordFrame と同じだが cl は閉じない)
        transition(m_composite.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        for (int i=0;i<(int)m_sims.size();++i)
        {
            Field& f=m_sims[i];
            transition(f.output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dl{}; dl.pResource=m_composite.Get(); dl.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dl.SubresourceIndex=0;
            D3D12_TEXTURE_COPY_LOCATION sl{}; sl.pResource=f.output.Get(); sl.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; sl.SubresourceIndex=0;
            m_cl->CopyTextureRegion(&dl, UINT(i*m_tileW), 0, 0, &sl, nullptr);
            transition(f.output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        // composite → backbuffer
        transition(m_back[bb].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        transition(m_composite.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_cl->CopyResource(m_back[bb].Get(), m_composite.Get());
        transition(m_back[bb].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
        // ImGui
        auto rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart(); rtv.ptr += SIZE_T(bb)*m_rtvInc;
        m_cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        ID3D12DescriptorHeap* ih[] = {m_imguiHeap.Get()}; m_cl->SetDescriptorHeaps(1, ih);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_cl.Get());
        transition(m_back[bb].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        m_cl->Close();
        submitAndWait();
        m_swap->Present(1, 0);
    }

    void buildShaders();   // 下で定義
    Dx12ComputeContext& makePass(const std::string& body)
    {
        m_passes.emplace_back();
        Dx12ComputeContext& p = m_passes.back();
        p.initialize(m_device.Get());
        std::string src = std::string(kCB) + body;
        if (!p.setShader(src.c_str(), "CSMain")) throw std::runtime_error("compute shader compile failed");
        return p;
    }

    // ── メンバ ─────────────────────────────────────────────────────────
    int m_gridN = 128;
    int m_imgW = 1280, m_imgH = 720, m_tileW = 1280;
    bool m_selftest = false;

    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_queue;
    ComPtr<ID3D12CommandAllocator> m_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cl;
    ComPtr<ID3D12Fence> m_fence; HANDLE m_fenceEvent=nullptr; UINT64 m_fenceVal=0;
    ComPtr<ID3D12DescriptorHeap> m_heap; UINT m_inc=0; UINT m_cursor=0;
    ComPtr<ID3D12Resource> m_composite;

    std::vector<Field> m_sims;

    std::vector<Dx12ComputeContext> m_passes;
    Dx12ComputeContext *m_advectVel=nullptr,*m_advectScalar=nullptr,*m_inject=nullptr,
                       *m_buoyancy=nullptr,*m_curlPass=nullptr,*m_vorticity=nullptr,
                       *m_divergence=nullptr,*m_clear=nullptr,*m_jacobi=nullptr,
                       *m_project=nullptr,*m_copyVec=nullptr,*m_copyScalar=nullptr,*m_raymarch=nullptr;

    // 窓
    HWND m_hwnd=nullptr;
    ComPtr<IDXGISwapChain3> m_swap;
    ComPtr<ID3D12Resource> m_back[3];
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap; UINT m_rtvInc=0;
    ComPtr<ID3D12DescriptorHeap> m_imguiHeap;
};

// ── compute pass (シェーダ) の生成 ───────────────────────────────────────
void VolSmoke::buildShaders()
{
    m_passes.reserve(16);

    m_advectVel = &makePass(R"HLSL(
RWTexture3D<float4> VelDst : register(u0);
RWTexture3D<float4> VelSrc : register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    float3 v=VelSrc[id].xyz;
    float4 nv; TRILERP(VelSrc, float3(id)-v*dt, nv);
    VelDst[id]=float4(nv.xyz*saturate(1.0-velDissip*dt),0);
})HLSL");

    m_advectScalar = &makePass(R"HLSL(
RWTexture3D<float>  DenDst:register(u0);
RWTexture3D<float>  TmpDst:register(u1);
RWTexture3D<float>  DenSrc:register(u2);
RWTexture3D<float>  TmpSrc:register(u3);
RWTexture3D<float4> VelSrc:register(u4);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    float3 v=VelSrc[id].xyz; float3 src=float3(id)-v*dt;
    float den; TRILERP(DenSrc, src, den);
    float tmp; TRILERP(TmpSrc, src, tmp);
    DenDst[id]=den*saturate(1.0-densDissip*dt);
    TmpDst[id]=tmp*saturate(1.0-cooling*dt);
})HLSL");

    m_inject = &makePass(R"HLSL(
RWTexture3D<float>  Den:register(u0);
RWTexture3D<float>  Tmp:register(u1);
RWTexture3D<float4> Vel:register(u2);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    float3 uvw=(float3(id)+0.5)/gridN;
    float r=length((uvw-emitPos)/emitRadius);
    float fall=saturate(1.0-r);
    if(fall>0){
        float k=fall;
        float n=frac(sin(dot(uvw, float3(91.7,113.5,47.3))*311.0 + time*5.0)*43758.5);
        float lump = 0.80 + 0.30*n;
        Tmp[id]=max(Tmp[id], emitTemp*k*lump);
        Den[id]=max(Den[id], emitDensity*k*lump);
        float4 v=Vel[id];
        v.y += emitVel*k*dt;
        Vel[id]=v;
    }
})HLSL");

    m_buoyancy = &makePass(R"HLSL(
RWTexture3D<float4> Vel:register(u0);
RWTexture3D<float>  Tmp:register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    float4 v=Vel[id];
    v.y += buoyancy*Tmp[id]*dt;
    v.xyz += wind*dt;
    Vel[id]=v;
})HLSL");

    m_curlPass = &makePass(R"HLSL(
RWTexture3D<float4> Curl:register(u0);
RWTexture3D<float4> Vel :register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    int3 p=int3(id);
    float3 vxp=Vel[clampi(p+int3(1,0,0))].xyz, vxn=Vel[clampi(p-int3(1,0,0))].xyz;
    float3 vyp=Vel[clampi(p+int3(0,1,0))].xyz, vyn=Vel[clampi(p-int3(0,1,0))].xyz;
    float3 vzp=Vel[clampi(p+int3(0,0,1))].xyz, vzn=Vel[clampi(p-int3(0,0,1))].xyz;
    float3 w=0.5*float3((vyp.z-vyn.z)-(vzp.y-vzn.y),
                        (vzp.x-vzn.x)-(vxp.z-vxn.z),
                        (vxp.y-vxn.y)-(vyp.x-vyn.x));
    Curl[id]=float4(w,0);
})HLSL");

    m_vorticity = &makePass(R"HLSL(
RWTexture3D<float4> Vel :register(u0);
RWTexture3D<float4> Curl:register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    int3 p=int3(id);
    float mxp=length(Curl[clampi(p+int3(1,0,0))].xyz), mxn=length(Curl[clampi(p-int3(1,0,0))].xyz);
    float myp=length(Curl[clampi(p+int3(0,1,0))].xyz), myn=length(Curl[clampi(p-int3(0,1,0))].xyz);
    float mzp=length(Curl[clampi(p+int3(0,0,1))].xyz), mzn=length(Curl[clampi(p-int3(0,0,1))].xyz);
    float3 grad=0.5*float3(mxp-mxn, myp-myn, mzp-mzn);
    float3 N=grad/(length(grad)+1e-5);
    float3 w=Curl[id].xyz;
    float3 force=vorticity*cross(N,w);
    float4 v=Vel[id]; v.xyz += force*dt; Vel[id]=v;
})HLSL");

    m_divergence = &makePass(R"HLSL(
RWTexture3D<float>  Div:register(u0);
RWTexture3D<float4> Vel:register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    int3 p=int3(id);
    float vxp=Vel[clampi(p+int3(1,0,0))].x, vxn=Vel[clampi(p-int3(1,0,0))].x;
    float vyp=Vel[clampi(p+int3(0,1,0))].y, vyn=Vel[clampi(p-int3(0,1,0))].y;
    float vzp=Vel[clampi(p+int3(0,0,1))].z, vzn=Vel[clampi(p-int3(0,0,1))].z;
    Div[id]=0.5*((vxp-vxn)+(vyp-vyn)+(vzp-vzn));
})HLSL");

    m_clear = &makePass(R"HLSL(
RWTexture3D<float> Dst:register(u0);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return; Dst[id]=0;
})HLSL");

    m_jacobi = &makePass(R"HLSL(
RWTexture3D<float> Dst:register(u0);
RWTexture3D<float> Src:register(u1);
RWTexture3D<float> Div:register(u2);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    int3 p=int3(id);
    float L=Src[clampi(p-int3(1,0,0))], R=Src[clampi(p+int3(1,0,0))];
    float D=Src[clampi(p-int3(0,1,0))], U=Src[clampi(p+int3(0,1,0))];
    float B=Src[clampi(p-int3(0,0,1))], F=Src[clampi(p+int3(0,0,1))];
    Dst[id]=(L+R+D+U+B+F - Div[id])/6.0;
})HLSL");

    m_project = &makePass(R"HLSL(
RWTexture3D<float4> Vel:register(u0);
RWTexture3D<float>  Prs:register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return;
    int3 p=int3(id);
    float L=Prs[clampi(p-int3(1,0,0))], R=Prs[clampi(p+int3(1,0,0))];
    float D=Prs[clampi(p-int3(0,1,0))], U=Prs[clampi(p+int3(0,1,0))];
    float B=Prs[clampi(p-int3(0,0,1))], F=Prs[clampi(p+int3(0,0,1))];
    float4 v=Vel[id];
    v.xyz -= 0.5*float3(R-L, U-D, F-B);
    Vel[id]=v;
})HLSL");

    m_copyVec = &makePass(R"HLSL(
RWTexture3D<float4> Dst:register(u0);
RWTexture3D<float4> Src:register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return; Dst[id]=Src[id];
})HLSL");

    m_copyScalar = &makePass(R"HLSL(
RWTexture3D<float> Dst:register(u0);
RWTexture3D<float> Src:register(u1);
[numthreads(8,8,8)] void CSMain(uint3 id:SV_DispatchThreadID){
    if(id.x>=gridN||id.y>=gridN||id.z>=gridN) return; Dst[id]=Src[id];
})HLSL");

    m_raymarch = &makePass(R"HLSL(
RWTexture2D<float4> Out:register(u0);
RWTexture3D<float>  Den:register(u1);
RWTexture3D<float>  Tmp:register(u2);
bool boxHit(float3 ro,float3 rd,out float t0,out float t1){
    float3 inv=1.0/rd; float3 a=(0.0-ro)*inv, b=(1.0-ro)*inv;
    float3 mn=min(a,b), mx=max(a,b);
    t0=max(max(mn.x,mn.y),mn.z); t1=min(min(mx.x,mx.y),mx.z);
    return t1>max(t0,0.0);
}
float3 fireColor(float tn){
    float3 c = float3(0,0,0);
    c = lerp(c, float3(0.95,0.06,0.01), smoothstep(0.05,0.32,tn));
    c = lerp(c, float3(1.0, 0.42,0.07), smoothstep(0.32,0.56,tn));
    c = lerp(c, float3(1.0, 0.80,0.34), smoothstep(0.56,0.80,tn));
    c = lerp(c, float3(1.0, 0.97,0.88), smoothstep(0.80,1.00,tn));
    return c;
}
float3 blueFlame(float tn){
    float3 c = float3(0,0,0);
    c = lerp(c, float3(0.05,0.13,0.50), smoothstep(0.05,0.30,tn));
    c = lerp(c, float3(0.10,0.45,1.00), smoothstep(0.30,0.55,tn));
    c = lerp(c, float3(0.55,0.85,1.00), smoothstep(0.55,0.80,tn));
    c = lerp(c, float3(0.92,0.97,1.00), smoothstep(0.80,1.00,tn));
    return c;
}
float3 flameColor(float tn){ return (fireMode>0.5) ? blueFlame(tn) : fireColor(tn); }
float lightTrans(float3 p){
    float3 L=normalize(light); const int LS=12; float ls=0.035; float dsum=0;
    [loop] for(int i=1;i<=LS;++i){
        float3 q=p+L*(ls*i);
        if(q.x<0||q.y<0||q.z<0||q.x>1||q.y>1||q.z>1) break;
        float dd; TRILERP(Den, q*gridN-0.5, dd); dsum+=dd;
    }
    return exp(-dsum*ls*26.0);
}
[numthreads(8,8,1)] void CSMain(uint3 id:SV_DispatchThreadID){
    if((int)id.x>=imgW||(int)id.y>=imgH) return;
    float2 ndc=float2((id.x+0.5)/imgW*2-1, 1-(id.y+0.5)/imgH*2);
    float3 fwd=normalize(camTar-camPos);
    float3 rgt=normalize(cross(float3(0,1,0),fwd));
    float3 up =cross(fwd,rgt);
    float3 rd =normalize(fwd + ndc.x*fovTan*aspect*rgt + ndc.y*fovTan*up);
    float3 ro =camPos;
    float3 bg=lerp(float3(0.015,0.018,0.026),float3(0.04,0.05,0.075),saturate(ndc.y*0.5+0.5));
    float3 acc=0; float trans=1.0;
    float t0,t1;
    if(boxHit(ro,rd,t0,t1)){
        const int STEPS=128; float dt=(t1-t0)/STEPS;
        for(int s=0;s<STEPS;++s){
            float t=t0+(s+0.5)*dt; float3 p=ro+rd*t;
            float3 gp=p*gridN-0.5;
            float d; TRILERP(Den, gp, d);
            float tp; TRILERP(Tmp, gp, tp);
            float tn=saturate(tp*tempScale);
            if(fireGain>0.0 && tn>0.02){
                float3 emis=flameColor(tn)*pow(tn,2.0)*fireGain;
                acc += trans*emis*dt;
            }
            if(d>0.002){
                float a=1.0-exp(-d*absorb*dt);
                float3 base=lerp(smokeColor*0.12, smokeColor, saturate(d*0.5));
                float sh=lightTrans(p);
                float3 lit=base*(ambient + 0.95*sh);
                acc += trans*a*lit; trans*=(1.0-a);
                if(trans<0.01) break;
            }
        }
    }
    float3 col=acc+trans*bg;
    col=1.0-exp(-col*1.2);
    Out[id.xy]=float4(col,1.0);
})HLSL");
}

static std::vector<std::string> splitComma(const std::string& s)
{
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c==',') { if(!cur.empty()) out.push_back(cur); cur.clear(); } else cur+=c; }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int main(int argc, char** argv)
{
    std::string out="frame.png", seq, preset="fire", compareList;
    int frames=1, grid=128, seqFrom=0, stride=1, width=1280, height=720;
    bool interactive=false, selftest=false, compare=false;
    for (int i=1;i<argc;++i)
    {
        std::string a=argv[i];
        if      (a=="--out"     && i+1<argc) out=argv[++i];
        else if (a=="--frames"  && i+1<argc) frames=std::atoi(argv[++i]);
        else if (a=="--grid"    && i+1<argc) grid=std::atoi(argv[++i]);
        else if (a=="--seq"     && i+1<argc) seq=argv[++i];
        else if (a=="--seqfrom" && i+1<argc) seqFrom=std::atoi(argv[++i]);
        else if (a=="--stride"  && i+1<argc) stride=std::atoi(argv[++i]);
        else if (a=="--preset"  && i+1<argc) preset=argv[++i];
        else if (a=="--width"   && i+1<argc) width=std::atoi(argv[++i]);
        else if (a=="--height"  && i+1<argc) height=std::atoi(argv[++i]);
        else if (a=="--interactive" || a=="-i") interactive=true;
        else if (a=="--selftest") { interactive=true; selftest=true; }
        else if (a=="--compare") { compare=true;
            if (i+1<argc && argv[i+1][0] != '-') compareList=argv[++i]; }
    }
    try
    {
        VolSmoke app;
        std::vector<std::string> presets =
            compare ? splitComma(compareList.empty() ? std::string("fire,blue,smoke") : compareList)
                    : std::vector<std::string>{ preset };
        if (interactive || selftest) app.runInteractive(grid, presets, width, height, selftest);
        else                         app.run(out, frames, grid, width, height, presets, seq, seqFrom, stride);
    }
    catch (const std::exception& e) { std::fprintf(stderr, "[volsmoke] ERROR: %s\n", e.what()); return 1; }
    return 0;
}
