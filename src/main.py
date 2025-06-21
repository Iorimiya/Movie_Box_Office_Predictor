from argparse import ArgumentParser, Namespace
import sys


from src.cli.argument_parser_builder import ArgumentParserBuilder


def main() -> None:
    """
    CLI 應用程式的主要入口點。

    它負責建構命令列解析器、解析輸入引數，
    並根據解析結果執行對應的命令處理函式。
    """
    builder: ArgumentParserBuilder = ArgumentParserBuilder()
    parser: ArgumentParser = builder.build()  # 建立完整的解析器

    # 解析命令列引數
    # argparse.parse_args() 會自動處理必填引數的檢查，
    # 如果缺少必填引數，會自動退出並顯示錯誤訊息。
    args: Namespace = parser.parse_args()

    # 檢查解析後的 'args' 物件是否包含 'func' 屬性
    # 'func' 屬性是我們透過 set_defaults() 綁定到各子命令的處理函式
    if hasattr(args, 'func'):
        try:
            # 由於我們在 handlers 類別的 __init__ 中傳遞了 parser 物件，
            # 處理函式內部可以直接使用 self.parser.error() 來報告錯誤。
            # 因此，這裡不需要再將 parser 物件額外附加到 args 上。

            # 執行綁定到命令的處理函式
            # args.func 已經是一個 bound method (例如 self.dataset_handler.create_index)，
            # 它會自動將 self 傳遞給方法，我們只需傳遞 args 物件即可。
            args.func(args)
        except SystemExit:
            # argparse.error() 會在內部呼叫 sys.exit()，
            # 因此捕獲 SystemExit 是為了避免未預期的額外錯誤訊息，
            # 因為 argparse 已經負責顯示了友善的錯誤提示。
            pass
        except Exception as e:
            # 捕獲並處理其他未預期的執行時錯誤
            print(f"命令執行失敗：{e}", file=sys.stderr)
            sys.exit(1)  # 以非零狀態碼退出，表示執行失敗
    else:
        # 如果使用者只執行了主程式名 (例如 `movie_predictor`) 而沒有提供任何子命令，
        # 則通常會顯示主解析器的幫助訊息。
        parser.print_help()
        sys.exit(1)  # 以非零狀態碼退出，表示命令使用不完整


if __name__ == "__main__":
    main()
