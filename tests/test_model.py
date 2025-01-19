def func_test_model(x):
    return x + 1

def test_answer():
    assert func_test_model(3) == 4
    # content of test_class.py
class TestClass:
    def test_shape(self):
        x = "this"
        assert "h" in x

    def test_(self):
        x = "hello"
        assert hasattr(x, "check")