#include <windows.h>
#include <stdio.h>
#include <aclapi.h>

// 关闭窗口的回调函数
BOOL WINAPI ConsoleHandler(DWORD signal) {
    switch (signal) {
        case CTRL_CLOSE_EVENT: {
            // 提示用户确认
            printf("您确定要关闭窗口吗？ (y/n): ");
            char response;
            scanf(" %c", &response);
            if (response == 'y' || response == 'Y') {
                return FALSE; // 允许关闭
            } else {
                printf("关闭操作被取消。\n");
                return TRUE; // 阻止关闭
            }
        }
        case CTRL_C_EVENT:
            printf("CTRL+C 被捕获，程序将继续运行。\n");
            return TRUE; // 阻止程序退出
        default:
            return FALSE; // 让系统处理其他信号
    }
}

// 窗口过程
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE: {
            // 提示用户确认
            printf("您确定要关闭窗口吗？ (y/n): ");
            char response;
            scanf(" %c", &response);
            if (response == 'y' || response == 'Y') {
                DestroyWindow(hwnd); // 允许关闭窗口
            }
            return 0; // 阻止默认处理
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

// 创建控制台窗口
void CreateConsoleWindow() {
    HWND hwndConsole = GetConsoleWindow();
    if (hwndConsole != NULL) {
        // 设置窗口过程
        SetWindowLongPtr(hwndConsole, GWLP_WNDPROC, (LONG_PTR)WindowProc);
    }
}

static const BOOL ProtectProcess()
{
    HANDLE hProcess = GetCurrentProcess();
    EXPLICIT_ACCESS denyAccess = {0};
    DWORD dwAccessPermissions = GENERIC_WRITE|PROCESS_ALL_ACCESS|WRITE_DAC|DELETE|WRITE_OWNER|READ_CONTROL;
    BuildExplicitAccessWithName( &denyAccess, L("CURRENT_USER"), dwAccessPermissions, DENY_ACCESS, NO_INHERITANCE );
    PACL pTempDacl = NULL;
    DWORD dwErr = 0;
    dwErr = SetEntriesInAcl( 1, &denyAccess, NULL, &pTempDacl );
    // check dwErr...
    dwErr = SetSecurityInfo( hProcess, SE_KERNEL_OBJECT, DACL_SECURITY_INFORMATION, NULL, NULL, pTempDacl, NULL );
    // check dwErr...
    LocalFree( pTempDacl );
    CloseHandle( hProcess );
    return dwErr == ERROR_SUCCESS;
}

int main() {
    SetConsoleOutputCP(CP_UTF8); 
    // 设置控制台关闭窗口回调
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    
    // 创建控制台窗口
    CreateConsoleWindow();
    ProtectProcess();
    printf("程序正在运行。尝试关闭窗口将提示确认。\n");
    
    // 程序主循环
    while (1) {
        Sleep(1000); // 模拟工作
    }

    return 0;
}
