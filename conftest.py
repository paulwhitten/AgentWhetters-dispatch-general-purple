def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:9009",
        help="URL of the running agent to test against",
    )
