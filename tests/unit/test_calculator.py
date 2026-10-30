from app.tools.calculator import calculate


def test_calculator_basic_and_functions():
    res = calculate.invoke({"expression": "2**10 + sqrt(16)"})
    assert res == "1028"

    res2 = calculate.invoke({"expression": "abs(-42) + 8 * 2"})
    assert res2 == "58"


def test_calculator_disallows_code_execution():
    res = calculate.invoke({"expression": "__import__('os').system('ls')"})
    assert res.startswith("ERROR:")

    res2 = calculate.invoke({"expression": "open('/etc/passwd').read()"})
    assert res2.startswith("ERROR:")


def test_calculator_disallows_huge_exponents():
    res = calculate.invoke({"expression": "2**1001"})
    assert res.startswith("ERROR:")


def test_calculator_rejects_nested_power_growth():
    """H-09: nested powers must be rejected before building a huge integer."""
    import time

    start = time.perf_counter()
    res = calculate.invoke({"expression": "((9**999)**999)**999"})
    elapsed = time.perf_counter() - start

    assert res.startswith("ERROR:")
    assert "too large" in res.lower()
    assert elapsed < 0.5


def test_calculator_handles_syntax_errors():
    res = calculate.invoke({"expression": "2 +* 3"})
    assert res.startswith("ERROR:")
