from mesh_router.app import _downstream_payload
from mesh_router.schemas import ChatCompletionRequest

def test_downstream_chat_defaults_stream_false():
    req = ChatCompletionRequest(model='m', messages=[{'role':'user','content':'x'}])
    assert _downstream_payload(req)['stream'] is False

def test_downstream_chat_preserves_stream_true():
    req = ChatCompletionRequest(model='m', messages=[{'role':'user','content':'x'}], stream=True)
    assert _downstream_payload(req)['stream'] is True
