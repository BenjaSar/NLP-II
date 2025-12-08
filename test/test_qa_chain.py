from langchain.chains import QAChain

def test_qa_chain_response():
    qa_chain = QAChain()
    response = qa_chain({"query": "What is the main topic?"})
    assert "result" in response
    assert len(response["result"]) > 0