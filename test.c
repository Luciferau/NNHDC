#include <stdio.h>
#include <windows.h> // 包含 Windows API 的头文件

int main() {
    // 获取当前进程的PID
    DWORD pid = GetCurrentProcessId();

    // 打印PID
    printf("current process pid is: %lu\n", pid);
    while(1){}
    
    return 0;
}
