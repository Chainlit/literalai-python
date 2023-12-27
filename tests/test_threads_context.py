from chainlit_client.client import ChainlitClient


def function_where_i_want_the_thread(client):
    return client.get_current_thread()


def test_thread_context():
    client = ChainlitClient()
    with client.thread() as thread_from_parent:
        thread_from_child = function_where_i_want_the_thread(client)

    assert thread_from_parent.id == thread_from_child.id
