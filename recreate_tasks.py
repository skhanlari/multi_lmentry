import argparse
from lmentry.constants import initialize_variables
from lmentry.constants import LANG, initialize_variables


def main():
    parser = argparse.ArgumentParser(prog="Recreate Tasks")
    parser.add_argument("-l", "--language", type=str, default=LANG)
    args = parser.parse_args()

    initialize_variables(args.language)

    from lmentry.tasks.lmentry_tasks import create_all_task_data

    create_all_task_data()


if __name__ == "__main__":
    main()
