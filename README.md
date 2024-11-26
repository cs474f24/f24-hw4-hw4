# HW 4: Image Denoising, Gibbs Sampling, and EM

There are 20 points possible for this assignment. The setup is worth 1 point,
the coding is worth 11 points, and the free-response questions are worth 8
points. The deadline for this assignment is posted on Canvas. Please read the
entire README before beginning the assignment.

## Partner policy and Academic integrity

This is a **partner** assignment. You may collaborate in any way with your
partner, but once you have selected your partner, all work you submit must
have been written by the two of you. You may *talk* with students other than
your partner about the concepts covered by the homework, but you may not share
code or answers with them in any way. If you have a question about an error
message or about why a numpy function returns what it does, post it on Piazza.
If you need help debugging your code, make a *private* post on Piazza or come
to office hours.

Finding a partner is your responsibility. If you need to find a partner, please
use the pinned "Search for Teammates!" post on Piazza. To submit an assignment
late by one day, each partner must spend one late day. You can choose to work
alone, but there is only one version of the assignment.

You may not reference existing implementations of the algorithms in this
assignment. We will use a combination of automated and manual methods for
comparing your code to that of other students. If we find sufficiently
suspicious similarities between your answers and those of another student or an
online source, you will be reported for a suspected violation. If you're unsure
of the academic integrity policies, ask for help; we can help you avoid
breaking the rules, but we can't un-report a suspected violation.

By pushing your code to GitHub, you agree to these rules, and understand that
there may be severe consequences for violating them.

## Important instructions

Your work will be graded and aggregated using an autograder that will download
the code and free response questions from each student's repository. If you
don't follow the instructions, you run the risk of getting *zero points*. The
`test_setup` test cases gives you points for following these instructions 
and will make it possible to grade your work easily.

The essential instructions:
- Your code must be *pushed* to GitHub for us to grade it!  We will only grade
  the latest version of your code that was pushed to GitHub before the
  deadline.
- You and your partner's NetIDs must be in the `netids` file; replace
  `NETID_GOES_HERE` with your netid, with one netid per line.
- Each of your free-response answers should be uploaded as a separate PDF to the
  [corresponding assignment on Canvas](
  https://canvas.northwestern.edu/courses/217080/assignments/1494574).

## Late Work

In general, late work is worth zero points. The autograder will only download
work from your repository that was pushed to GitHub before the deadline.
However:
- Each student gets three late days to use across the entire quarter. To turn
  this assignment in late by one day, each partner must use one late day.
  To use those late days, update the [HW4 Late Days assignment on Canvas](
  https://canvas.northwestern.edu/courses/217080/assignments/1494525) with a
  single integer representing the number of late days you wish to use.
- Late days apply equally to coding and free-response; using one late day means
  you can turn both the code and the free-response questions one day late.
- If you have a personal emergency, please ask for help. You do not have to
  share any personal information with me, but I will ask you to get in touch
  with the dean who oversees your student services to coordinate
  accommodations.

## Clone this repository

First, you need to clone this repository. As soon as you've done so, go ahead
and add your NetID(s) to the `netids` file, run `git add netid`, then `git
commit -m "added netid"`, and `git push origin main`.  If you've successfully
run those commands, you're done with the `test_setup` test case.

## Environment setup

Check HW2 for the details on setting up your virtual environment. Note that the
`requirements.txt` file has changed -- we need `pillow` for image processing.

## What to do for this assignment

In every function in `src/` where you need to write code, there is a `raise
NotImplementedError` which you will replace with your implementation. The test
cases will guide you through the work you need to do and tell you how many
points you've earned.  **We recommend that you try to write code to pass these
tests in the order they appear in `tests/rubric.json`.** That file will also
tell you how these tests depend on one another and how much each test is worth.
The test cases can be run from the root directory of this repository with:

``python -m pytest -s``

To run a single test, you can specify it with `-k`, e.g., `python -m pytest -s
-k test_setup`.  To run a group of tests, you can use `-k` with a prefix, e.g.,
`python -m pytest -s -k test_ising` will run all tests that begin with
`test_ising`.  The `-s` means that any print statements you include will in
fact be printed; the default behavior (`python -m pytest`) will suppress
everything but the pytest output.

We will use these test cases to grade your work! Even if you change the test
cases such that you pass the tests on your computer, we're still going to use
the original test cases to grade your assignment.

## Questions? Problems? Issues?

Ask a question on Piazza, and we'll help you there.
