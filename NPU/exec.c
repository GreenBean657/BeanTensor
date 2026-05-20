
int _unused_backup() {
    constexpr int N = 1024;  // must be multiple of 16

    // Allocate and fill input buffers
    std::vector<uint16_t> a(N), b(N), c(N), out(N);

    for (int i = 0; i < N; i++) {
        a[i]   = f32_to_bf16(1.0f);   // a = 1.0
        b[i]   = f32_to_bf16(2.0f);   // b = 2.0
        c[i]   = f32_to_bf16(0.5f);   // c = 0.5
        // expected out[i] = 1.0 * 2.0 + 0.5 = 2.5
    }

    if (!NPUBackend::is_available()) {
        std::cerr << "No XDNA2 NPU detected\n";
        return 1;
    }

    NPUBackend npu;
    npu.fma_bf16(a.data(), b.data(), c.data(), out.data(), N);

    // Verify first few results
    bool pass = true;
    for (int i = 0; i < 8; i++) {
        float result   = bf16_to_f32(out[i]);
        float expected = 2.5f;
        std::cout << "out[" << i << "] = " << result
                  << " (expected " << expected << ")\n";
        if (std::abs(result - expected) > 0.01f) pass = false;
    }

    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
