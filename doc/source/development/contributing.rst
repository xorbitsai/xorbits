.. _contributing:

***********************
Contributing to Xorbits
***********************

.. contents:: Table of contents:
   :local:

Where to start?
===============

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

If you are brand new to Xorbits or open-source development, we recommend going
through the `GitHub "issues" tab <https://github.com/xorbitsai/xorbits/issues>`_
to find issues that interest you. There are a number of issues listed under `Docs
<https://github.com/xorbitsai/xorbits/issues?labels=documentation&sort=updated&state=open>`_
and `good first issue
<https://github.com/xorbitsai/xorbits/issues?labels=good+first+issue&sort=updated&state=open>`_
where you could start out. Once you've found an interesting issue, you can
return here to get your development environment setup.

When you start working on an issue, it's a good idea to assign the issue to yourself,
so nobody else duplicates the work on it. GitHub restricts assigning issues to maintainers
of the project only. In most projects, and until recently in Xorbits, contributors added a
comment letting others know they are working on an issue. While this is ok, you need to
check each issue individually, and it's not possible to find the unassigned ones.

For this reason, we implemented a workaround consisting of adding a comment with the exact
text ``take``. When you do it, a GitHub action will automatically assign you the issue
(this will take seconds, and may require refreshing the page to see it).
By doing this, it's possible to filter the list of issues and find only the unassigned ones.

So, a good way to find an issue to start contributing to Xorbits is to check the list of
`unassigned good first issues <https://github.com/xorbitsai/xorbits/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22+no%3Aassignee>`_
and assign yourself one you like by writing a comment with the exact text ``take``.

If for whatever reason you are not able to continue working with the issue, please try to
unassign it, so other people know it's available again. You can check the list of
assigned issues, since people may not be working in them anymore. If you want to work on one
that is assigned, feel free to kindly ask the current assignee if you can take it
(please allow at least a week of inactivity before considering work in the issue discontinued).

We have several `contributor community <https://xorbits.io/community>`_ communication channels, which you are
welcome to join, and ask questions as you figure things out. Among them are regular meetings for
new contributors, dev meetings, a discourse forum, and a slack for the contributor community.
All Xorbits contributors are welcome to these spaces, where they can connect with each other. Even
maintainers who have been with us for a long time felt just like you when they started out, and
are happy to welcome you and support you as you get to know how we work, and where things are.
Take a look at the next sections to learn more.

.. _contributing.bug_reports:

Bug reports and enhancement requests
====================================

Bug reports are an important part of making Xorbits more stable. Having a complete bug report
will allow others to reproduce the bug and provide insight into fixing. See
`this stackoverflow article <https://stackoverflow.com/help/mcve>`_ and
`this blogpost <https://matthewrocklin.com/minimal-bug-reports>`_
for tips on writing a good bug report.

Trying the bug-producing code out on the *main* branch is often a worthwhile exercise
to confirm the bug still exists. It is also worth searching existing bug reports and pull requests
to see if the issue has already been reported and/or fixed.

Bug reports must:

#. Include a short, self-contained Python snippet reproducing the problem.
   You can format the code nicely by using `GitHub Flavored Markdown
   <https://github.github.com/github-flavored-markdown/>`_::

      ```python
      >>> from xorbits.pandas import DataFrame
      >>> df = DataFrame(...)
      ...
      ```

#. Include the full version string of Xorbits and its dependencies.

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the Xorbits community and be open to comments/ideas from others.

.. _contributing.github:

Working with the code
=====================

Now that you have an issue you want to fix, enhancement to add, or documentation to improve,
you need to learn how to work with GitHub and the Xorbits code base.

.. _contributing.version_control:

Version control, Git, and GitHub
--------------------------------

To the new user, working with Git is one of the more daunting aspects of contributing to Xorbits.
It can very quickly become overwhelming, but sticking to the guidelines below will help keep the process
straightforward and mostly trouble free.  As always, if you are having difficulties please
feel free to ask for help.

The code is hosted on `GitHub <https://www.github.com/xorbitsai/xorbits>`_. To
contribute you will need to sign up for a `free GitHub account
<https://github.com/signup/free>`_. We use `Git <https://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning Git:

* the `GitHub help pages <https://help.github.com/>`_.
* the `NumPy documentation <https://numpy.org/doc/stable/dev/index.html>`_.
* Matthew Brett's `Pydagogue <https://matthew-brett.github.io/pydagogue/>`_.

Getting started with Git
------------------------

`GitHub has instructions <https://help.github.com/set-up-git-redirect>`__ for installing git,
setting up your SSH key, and configuring git. All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

.. _contributing.forking:

Forking
-------

You will need your own fork to work on the code. Go to the `Xorbits project
page <https://github.com/xorbitsai/xorbits>`_ and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git clone --recursive https://github.com/your-user-name/xorbits.git xorbits-yourname
    cd xorbits-yourname
    git remote add upstream https://github.com/xorbitsai/xorbits.git

This creates the directory ``xorbits-yourname`` and connects your repository to
the upstream (main project) *Xorbits* repository.

Note that performing a shallow clone (with ``--depth==N``, for some ``N`` greater
or equal to 1) might break some tests.

Creating a branch
-----------------

You want your main branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch. Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to Xorbits. You can have many shiny-new-features
and switch in between them using the git checkout command.

When creating this branch, make sure your main branch is up to date with
the latest upstream main version. To update your local main branch, you
can do::

    git checkout main
    git pull upstream main --ff-only

When you want to update the feature branch with changes in main after
you created the branch, check the section on
:ref:`updating a PR <contributing.update-pr>`.

Contributing your changes to Xorbits
====================================

.. _contributing.commit-code:

Committing your code
--------------------

Keep style fixes to a separate commit to make your pull request more readable.

Once you've made changes, you can see them by typing::

    git status

If you have created a new file, it is not being tracked by git. Add it by typing::

    git add path/to/file-to-be-added.py

Doing 'git status' again should give something like::

    # On branch shiny-new-feature
    #
    #       modified:   /relative/path/to/file-you-added.py
    #

Finally, commit your changes to your local repository with an explanatory message. Xorbits
uses a convention for commit message prefixes and layout.  Here are
some common prefixes along with general guidelines for when to use them:

* FEATURE: Feature, new functionality
* ENH: Enhancement
* BUG: Bug fix
* DOC: Additions/updates to documentation
* TST: Additions/updates to tests
* BLD: Updates to the build process/scripts
* PERF: Performance improvement
* TYP: Type annotations
* CLN: Code cleanup
* REF: Refactor
* CHORE: chore

The following defines how a commit message should be structured.  Please reference the
relevant GitHub issues in your commit message using GH1234 or #1234.  Either style
is fine, but the former is generally preferred:

* a subject line with ``< 80`` chars.
* One blank line.
* Optionally, a commit message body.

Now you can commit your changes in your local repository::

    git commit -m

.. _contributing.push-code:

Pushing your changes
--------------------

When you want your changes to appear publicly on your GitHub page, push your
forked feature branch's commits::

    git push origin shiny-new-feature

Here ``origin`` is the default name given to your remote repository on GitHub.
You can see the remote repositories::

    git remote -v

If you added the upstream repository as described above you will see something
like::

    origin  git@github.com:yourname/xorbits.git (fetch)
    origin  git@github.com:yourname/xorbits.git (push)
    upstream        git://github.com/xorbitsai/xorbits.git (fetch)
    upstream        git://github.com/xorbitsai/xorbits.git (push)

Now your code is on GitHub, but it is not yet a part of the Xorbits project. For that to
happen, a pull request needs to be submitted on GitHub.

Review your code
----------------

When you're ready to ask for a code review, file a pull request. Before you do, once
again make sure that you have followed all the guidelines outlined in this document
regarding code style, tests, performance tests, and documentation. You should also
double check your branch changes against the branch it was based on:

#. Navigate to your repository on GitHub -- https://github.com/your-user-name/xorbits
#. Click on ``Branches``
#. Click on the ``Compare`` button for your feature branch
#. Select the ``base`` and ``compare`` branches, if necessary. This will be ``main`` and
   ``shiny-new-feature``, respectively.

Finally, make the pull request
------------------------------

If everything looks good, you are ready to make a pull request.  A pull request is how
code from a local repository becomes available to the GitHub community and can be looked
at and eventually merged into the main version.  This pull request and its associated
changes will eventually be committed to the main branch and available in the next
release.  To submit a pull request:

#. Navigate to your repository on GitHub
#. Click on the ``Pull Request`` button
#. You can then click on ``Commits`` and ``Files Changed`` to make sure everything looks
   okay one last time
#. Write a description of your changes in the ``Preview Discussion`` tab
#. Click ``Send Pull Request``.

This request then goes to the repository maintainers, and they will review
the code.

.. _contributing.update-pr:

Updating your pull request
--------------------------

Based on the review you get on your pull request, you will probably need to make
some changes to the code. In that case, you can make them in your branch,
add a new commit to that branch, push it to GitHub, and the pull request will be
automatically updated.  Pushing them to GitHub again is done by::

    git push origin shiny-new-feature

This will automatically update your pull request with the latest code and restart the
:any:`Continuous Integration <contributing.ci>` tests.

Another reason you might need to update your pull request is to solve conflicts
with changes that have been merged into the main branch since you opened your
pull request.

To do this, you need to "merge upstream main" in your branch::

    git checkout shiny-new-feature
    git fetch upstream
    git merge upstream/main

If there are no conflicts (or they could be fixed automatically), a file with a
default commit message will open, and you can simply save and quit this file.

If there are merge conflicts, you need to solve those conflicts. See for
example at https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/
for an explanation on how to do this.
Once the conflicts are merged and the files where the conflicts were solved are
added, you can run ``git commit`` to save those fixes.

If you have uncommitted changes at the moment you want to update the branch with
main, you will need to ``stash`` them prior to updating (see the
`stash docs <https://git-scm.com/book/en/v2/Git-Tools-Stashing-and-Cleaning>`__).
This will effectively store your changes and they can be reapplied after updating.

After the feature branch has been update locally, you can now update your pull
request by pushing to the branch on GitHub::

    git push origin shiny-new-feature

Autofixing formatting errors
----------------------------

We use several styling checks (e.g. ``black``, ``flake8``, ``isort``) which are run after
you make a pull request.

To automatically fix formatting errors on each commit you make, you can
set up pre-commit yourself. First, create a Python :ref:`environment
<contributing_environment>` and then set up :ref:`pre-commit <contributing.pre-commit>`.

Delete your merged branch (optional)
------------------------------------

Once your feature branch is accepted into upstream, you'll probably want to get rid of
the branch. First, merge upstream main into your branch so git knows it is safe to
delete your branch::

    git fetch upstream
    git checkout main
    git merge upstream/main

Then you can do::

    git branch -d shiny-new-feature

Make sure you use a lower-case ``-d``, or else git won't warn you if your feature
branch has not actually been merged.

The branch will still exist on GitHub, so to delete it there do::

    git push origin --delete shiny-new-feature


Tips for a successful pull request
==================================

If you have made it to the `Review your code`_ phase, one of the core contributors may
take a look. Please note however that a handful of people are responsible for reviewing
all of the contributions, which can often lead to bottlenecks.

To improve the chances of your pull request being reviewed, you should:

- **Reference an open issue** for non-trivial changes to clarify the PR's purpose
- **Ensure you have appropriate tests**. These should be the first part of any PR
- **Keep your pull requests as simple as possible**. Larger PRs take longer to review
- **Ensure that CI is in a green state**. Reviewers may not even look otherwise
- **Keep** `Updating your pull request`_, either by request or every few days
