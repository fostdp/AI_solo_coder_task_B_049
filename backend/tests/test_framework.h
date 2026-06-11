#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <cmath>
#include <iomanip>

namespace tcm_test {

struct TestCase {
    std::string name;
    std::string category;
    std::function<bool()> run;
};

struct TestResult {
    std::string name;
    std::string category;
    bool passed;
    std::string message;
    double elapsed_ms;
};

class TestRunner {
public:
    static TestRunner& instance() {
        static TestRunner r;
        return r;
    }

    void add_test(const std::string& category, const std::string& name, std::function<bool()> fn) {
        tests_.push_back({name, category, fn});
    }

    int run_all() {
        std::vector<TestResult> results;
        int pass = 0, fail = 0;
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           TCM 高级分析模块 · 单元测试套件                    ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

        std::string current_category;
        for (auto& t : tests_) {
            if (t.category != current_category) {
                current_category = t.category;
                std::cout << "\n━━━  " << current_category << "  ━━━\n";
            }
            auto start = std::chrono::high_resolution_clock::now();
            bool ok = false;
            std::string msg;
            try {
                ok = t.run();
                msg = ok ? "" : "assertion failed";
            } catch (const std::exception& e) {
                msg = std::string("exception: ") + e.what();
            } catch (...) {
                msg = "unknown exception";
            }
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            results.push_back({t.name, t.category, ok, msg, ms});
            std::cout << (ok ? "  ✅ " : "  ❌ ") << std::left << std::setw(48) << t.name
                      << "  [" << std::fixed << std::setprecision(2) << ms << "ms]";
            if (!ok) std::cout << "  ← " << msg;
            std::cout << "\n";
            ok ? pass++ : fail++;
        }

        std::cout << "\n═══════════════════════════════════════════════════════════════\n";
        std::cout << " 总计: " << tests_.size() << "  通过: " << pass << "  失败: " << fail
                  << "  通过率: " << std::fixed << std::setprecision(1)
                  << (tests_.empty() ? 0.0 : 100.0 * pass / tests_.size()) << "%\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        return fail == 0 ? 0 : 1;
    }

private:
    std::vector<TestCase> tests_;
};

#define TEST_CAT(cat, name) \
    static bool tcm_test_##cat##_##name(); \
    namespace { \
        struct tcm_test_reg_##cat##_##name { \
            tcm_test_reg_##cat##_##name() { \
                TestRunner::instance().add_test(#cat, #name, tcm_test_##cat##_##name); \
            } \
        } tcm_test_reg_inst_##cat##_##name; \
    } \
    static bool tcm_test_##cat##_##name()

inline bool approx_eq(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

} // namespace tcm_test
