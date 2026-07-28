import os


def main() -> None:
    print("**** Mock agent execution")
    os.chdir("/repository")

    print("0. Create new mock file LLM.md in current directory")
    os.system('echo "Mock agent" >> LLM.md')

    print("1. List files")
    os.system("ls -la")

    print("**** Mock agent execution finished!")


if __name__ == "__main__":
    main()
