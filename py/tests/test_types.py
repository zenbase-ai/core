from zenbase.types import LMDemo, deflm


def test_demo_eq():
    demoset = [
        LMDemo(inputs={}, outputs={"output": "a"}),
        LMDemo(inputs={}, outputs={"output": "b"}),
    ]

    # Structural inequality
    assert demoset[0] != demoset[1]
    # Structural equality
    assert demoset[0] == LMDemo(inputs={}, outputs={"output": "a"})


def test_lm_function_refine():
    fn = deflm(lambda r: r.inputs)
    assert fn != fn.clean_and_duplicate()
