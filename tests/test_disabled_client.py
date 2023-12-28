from chainlit_client import ChainlitClient


def test_disabled_client():
    client = ChainlitClient(enabled=False)
    client.instrument_openai()
    client.api.create_step()
    client.thread()
    client.get_current_step()


def test_disabled_client_decorator():
    client = ChainlitClient(enabled=False)

    @client.step
    def inner(n):
        return n

    @client.thread
    def foo(n):
        return 1 + inner(n)

    assert foo(1) == 2


def test_disabled_client_context():
    client = ChainlitClient(enabled=False)

    c = 1
    with client.step(type="foo"):
        a = 1 + c

    assert a == 2

    with client.thread(tags=["foo"]):
        b = 1 + c

    assert b == 2
