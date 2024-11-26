import re


def test_setup():
    """
    To pass this test case, you need to add your NetID(s) to the `netids` file
    in the root directory of your repository.
    """

    with open('netids', 'r') as inf:
        lines = inf.readlines()

    assert len(lines) <= 2, "At most two netids"

    dummy_netids = ["NETID_GOES_HERE", "ONE_NETID_PER_LINE"]
    for line in lines:
        netid = str(line.strip())
        msg = "Add your NetID(s) and delete placeholders"
        assert netid not in dummy_netids, msg
        msg = "Your NetID looks like xyz0123"
        assert re.search(r"^[a-z]{3}[0-9]{3,4}$", netid.lower()) is not None, msg
