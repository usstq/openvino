// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <tuple>
#include <memory>

#include <ie_compound_blob.h>

#include <gpu/gpu_config.hpp>
#include <common_test_utils/test_common.hpp>
#include <common_test_utils/test_constants.hpp>
#include "common_test_utils/file_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include <openvino/core/preprocess/pre_post_process.hpp>

#ifdef _WIN32
#ifdef  ENABLE_DX11

#ifndef D3D11_NO_HELPERS
#define D3D11_NO_HELPERS
#define D3D11_NO_HELPERS_DEFINED_CTX_UT
#endif

#ifndef NOMINMAX
#define NOMINMAX
#define NOMINMAX_DEFINED_CTX_UT
#endif

#include <gpu/gpu_context_api_dx.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#include <atlbase.h>
#include <d3d11.h>
#include <d3d11_4.h>

#ifdef NOMINMAX_DEFINED_CTX_UT
#undef NOMINMAX
#undef NOMINMAX_DEFINED_CTX_UT
#endif

#ifdef D3D11_NO_HELPERS_DEFINED_CTX_UT
#undef D3D11_NO_HELPERS
#undef D3D11_NO_HELPERS_DEFINED_CTX_UT
#endif

using namespace ::testing;

struct DX11RemoteCtx_Test : public CommonTestUtils::TestsCommon {
    virtual ~DX11RemoteCtx_Test() = default;

protected:
    CComPtr<IDXGIFactory> factory;
    std::vector<CComPtr<IDXGIAdapter>> intel_adapters;
    std::vector<CComPtr<IDXGIAdapter>> other_adapters;

    void SetUp() override {
        IDXGIFactory* out_factory = nullptr;
        HRESULT err = CreateDXGIFactory(__uuidof(IDXGIFactory),
                                        reinterpret_cast<void**>(&out_factory));
        if (FAILED(err)) {
            throw std::runtime_error("Cannot create CreateDXGIFactory, error: " + std::to_string(HRESULT_CODE(err)));
        }

        factory.Attach(out_factory);

        UINT adapter_index = 0;
        const unsigned int refIntelVendorID = 0x8086;
        IDXGIAdapter* out_adapter = nullptr;
        while (factory->EnumAdapters(adapter_index, &out_adapter) != DXGI_ERROR_NOT_FOUND) {
            CComPtr<IDXGIAdapter> adapter(out_adapter);

            DXGI_ADAPTER_DESC desc{};
            adapter->GetDesc(&desc);
            if (desc.VendorId == refIntelVendorID) {
                intel_adapters.push_back(adapter);
            } else {
                other_adapters.push_back(adapter);
            }
            ++adapter_index;
        }
    }

    std::tuple<CComPtr<ID3D11Device>, CComPtr<ID3D11DeviceContext>>
    create_device_with_ctx(CComPtr<IDXGIAdapter> adapter) {
        UINT flags = 0;
        D3D_FEATURE_LEVEL feature_levels[] = { D3D_FEATURE_LEVEL_11_1,
                                               D3D_FEATURE_LEVEL_11_0,
                                             };
        D3D_FEATURE_LEVEL featureLevel;
        ID3D11Device* ret_device_ptr = nullptr;
        ID3D11DeviceContext* ret_ctx_ptr = nullptr;
        HRESULT err = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN,
                                        nullptr, flags,
                                        feature_levels,
                                        ARRAYSIZE(feature_levels),
                                        D3D11_SDK_VERSION, &ret_device_ptr,
                                        &featureLevel, &ret_ctx_ptr);
        if (FAILED(err)) {
            throw std::runtime_error("Cannot create D3D11CreateDevice, error: " +
                                     std::to_string(HRESULT_CODE(err)));
        }

        return std::make_tuple(ret_device_ptr, ret_ctx_ptr);
    }
};

struct DX11CachedTexture_Test : DX11RemoteCtx_Test {
    D3D11_TEXTURE2D_DESC texture_description = { 0 };
    std::vector<CComPtr<ID3D11Texture2D>> dx11_textures;
    CComPtr<ID3D11Device> device_ptr;
    CComPtr<ID3D11DeviceContext> ctx_ptr;

    void SetUp() override {
        DX11RemoteCtx_Test::SetUp();
        ASSERT_FALSE(intel_adapters.empty());
        ASSERT_NO_THROW(std::tie(device_ptr, ctx_ptr) =
                        create_device_with_ctx(*intel_adapters.begin()));

        // create textures
        const size_t textures_count = 4;

        texture_description.Width = 1024;
        texture_description.Height = 768;
        texture_description.MipLevels = 1;

        texture_description.ArraySize = 1;
        texture_description.Format = DXGI_FORMAT_NV12;
        texture_description.SampleDesc.Count = 1;
        texture_description.Usage = D3D11_USAGE_DEFAULT;
        texture_description.MiscFlags = 0;
        texture_description.BindFlags = 0;

        dx11_textures.reserve(textures_count);
        HRESULT err = S_OK;
        for (size_t i = 0; i < textures_count; i++) {
            ID3D11Texture2D *pTexture2D = nullptr;
            err = device_ptr->CreateTexture2D(&texture_description, nullptr, &pTexture2D);
            ASSERT_FALSE(FAILED(err));
            dx11_textures.emplace_back(pTexture2D);
        }
    }

    void run_make_shared_nv12_tensor_cached_inference(bool is_caching_test) {
    #if defined(ANDROID)
        GTEST_SKIP();
    #endif
        // inference using remote blob with batch
        auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, texture_description.Height, texture_description.Width});
        ov::Core core;
        ov::intel_gpu::ocl::D3DContext context(core, device_ptr);

        using namespace ov::preprocess;
        auto p = PrePostProcessor(fn_ptr_remote);
        p.input().tensor().set_element_type(ov::element::u8)
                        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                        .set_memory_type(GPU_CONFIG_KEY(SURFACE));
        p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
        p.input().model().set_layout("NCHW");
        auto model = p.build();

        auto param_input_y = model->get_parameters().at(0);
        auto param_input_uv = model->get_parameters().at(1);

        const size_t total_run_number = 4;

        std::string cacheDirName;
        if (is_caching_test) {
            cacheDirName = std::string("make_shared_nv12_tensor_cached_inference");
            CommonTestUtils::removeFilesWithExt(cacheDirName, "blob");
            CommonTestUtils::removeFilesWithExt(cacheDirName, "cl_cache");
            CommonTestUtils::removeDir(cacheDirName);
            core.set_property(ov::cache_dir(cacheDirName));

            auto tmp_model = core.compile_model(model, context);
        }

        auto compiled_model = core.compile_model(model, context);
        auto request = compiled_model.create_infer_request();

        const size_t iteration_count = 10;
        for (size_t i = 0; i < iteration_count; i++) {
            auto tensor = context.create_tensor_nv12(texture_description.Height, texture_description.Width, dx11_textures[0]);
            request.set_tensor(param_input_y, tensor.first);
            request.set_tensor(param_input_uv, tensor.second);

            ASSERT_NO_THROW(request.infer());
            auto output_tensor = request.get_tensor(model->get_results().at(0));
        }

        if (is_caching_test) {
            CommonTestUtils::removeFilesWithExt(cacheDirName, "blob");
            CommonTestUtils::removeFilesWithExt(cacheDirName, "cl_cache");
            CommonTestUtils::removeDir(cacheDirName);
        }
    }
};

TEST_F(DX11RemoteCtx_Test, smoke_make_shared_context) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    using namespace InferenceEngine;
    using namespace InferenceEngine::gpu;
    auto ie = InferenceEngine::Core();

    CComPtr<ID3D11Device> device_ptr;
    CComPtr<ID3D11DeviceContext> ctx_ptr;

    ASSERT_NO_THROW(std::tie(device_ptr, ctx_ptr) =
        create_device_with_ctx(intel_adapters[0]));
    auto remote_context = make_shared_context(ie,
        CommonTestUtils::DEVICE_GPU,
        device_ptr);
    ASSERT_TRUE(remote_context);

    for (auto adapter : other_adapters) {
        CComPtr<ID3D11Device> device_ptr;
        CComPtr<ID3D11DeviceContext> ctx_ptr;

        ASSERT_NO_THROW(std::tie(device_ptr, ctx_ptr) =
                        create_device_with_ctx(adapter));
        ASSERT_THROW(make_shared_context(ie, CommonTestUtils::DEVICE_GPU,
                                         device_ptr),
                     std::runtime_error);
    }
}


TEST_F(DX11CachedTexture_Test, smoke_make_shared_nv12_blob_cached) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    using namespace InferenceEngine;
    using namespace InferenceEngine::gpu;
    auto ie = InferenceEngine::Core();
    auto remote_context = make_shared_context(ie, CommonTestUtils::DEVICE_GPU,
                                                  device_ptr);
    ASSERT_TRUE(remote_context);
    const size_t total_run_number = 4;
    for (size_t i = 0; i < total_run_number; i++) {
        for (const auto& t : dx11_textures) {
            auto blob = make_shared_blob_nv12(texture_description.Height,
                                              texture_description.Width,
                                              remote_context, t);
            ASSERT_TRUE(blob);
            ASSERT_NO_THROW(blob->allocate());
        }
    }
}

TEST_F(DX11CachedTexture_Test, _make_shared_nv12_blob_cached_inference) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    using namespace InferenceEngine;
    using namespace InferenceEngine::gpu;
    // inference using remote blob with batch
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, texture_description.Height, texture_description.Width});
    auto ie = InferenceEngine::Core();

    CNNNetwork net(fn_ptr_remote);
    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(ColorFormat::NV12);

    auto remote_context = make_shared_context(ie, CommonTestUtils::DEVICE_GPU, device_ptr);
    Blob::Ptr nv12_blob = make_shared_blob_nv12(texture_description.Height,
                                            texture_description.Width,
                                            remote_context, dx11_textures[0]);

    ASSERT_TRUE(remote_context);
    const size_t total_run_number = 4;

    {
        auto exec_net = ie.LoadNetwork(net, remote_context,
            { { GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS, PluginConfigParams::YES} });

        // inference using shared nv12 blob
        auto inf_req_shared = exec_net.CreateInferRequest();
        auto dims = net.getInputsInfo().begin()->second->getTensorDesc().getDims();
        size_t imSize = dims[1] * dims[2] * dims[3];

        const size_t iteration_count = 10;
        for (size_t i = 0; i < iteration_count; i++) {
            inf_req_shared.SetBlob(net.getInputsInfo().begin()->first, nv12_blob);

            inf_req_shared.Infer();
            auto outputBlob_shared = inf_req_shared.GetBlob(net.getOutputsInfo().begin()->first);
        }
    }
}

TEST_F(DX11CachedTexture_Test, smoke_make_shared_nv12_tensor_cached) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    ov::Core core;
    ov::intel_gpu::ocl::D3DContext context(core, device_ptr);
    const size_t total_run_number = 4;
    for (size_t i = 0; i < total_run_number; i++) {
        for (const auto& t : dx11_textures) {
            ASSERT_NO_THROW(auto tensor = context.create_tensor_nv12(texture_description.Height, texture_description.Width, t));
        }
    }
}

TEST_F(DX11CachedTexture_Test, _make_shared_nv12_tensor_cached_inference) {
    this->run_make_shared_nv12_tensor_cached_inference(false);
}

TEST_F(DX11CachedTexture_Test, _make_shared_nv12_tensor_cached_inference_cached) {
    this->run_make_shared_nv12_tensor_cached_inference(true);
}

#endif // ENABLE_DX11
#endif // WIN32
