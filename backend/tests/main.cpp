#include "test_framework.h"

#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    int code = tcm_test::TestRunner::instance().run_all();
    std::cout << (code == 0 ? "\n🎉 全部测试通过\n" : "\n💥 存在测试失败\n") << std::endl;
    return code;
}
