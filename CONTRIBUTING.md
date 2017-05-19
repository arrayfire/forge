# Contributing to Forge

We'd love for you to contribute to our source code and to make Forge even better than it is
today! Here are the guidelines we'd like you to follow:

 - [Question or Problem?](#question)
 - [Issues and Bugs](#issue)
 - [Feature Requests](#feature)
 - [Submission Guidelines](#submit)
 - [Commit Message Guidelines](#commit)

## <a name="question"></a> Got a Question or Problem?

If you have questions about how to use Forge, please direct these to the [Google Group][groups]
discussion list or [Github][github]. We are also available on [Gitter][gitter].

## <a name="issue"></a> Found an Issue?

If you find a bug in the source code or a mistake in the documentation, you can help us by
submitting an issue to our [GitHub Repository][github]. Even better you can submit a Pull Request
with a fix.

## <a name="feature"></a> Want a Feature?

You can request a new feature by submitting an issue to our [GitHub Repository][github].  If you
would like to implement a new feature then consider what kind of change it is:

* **Major Changes** that you wish to contribute to the project should be discussed first on our
  [Github][github] or [Gitter][gitter] so that we can better coordinate our efforts,
  prevent duplication of work, and help you to craft the change so that it is successfully accepted
  into the project.
* **Small Changes** can be crafted and submitted to the [GitHub Repository][github] as a Pull
  Request.


## <a name="docs"></a> Want a Documentation Fix?

If you want to help improve the documentation, it's a good idea to let others know what you're working on to
minimize duplication of effort. Create a new issue (or comment on a related existing one) to let
others know what you're working on.

Please build and test the documentation before submitting the PR to be sure you
haven't accidentally introduced any layout or formatting issues. You should also make sure that your
commit message starts with "docs" and follows the **[Commit Message Guidelines](#commit)** outlined
below.

## <a name="submit"></a> Submission Guidelines

### Submitting an Issue
Before you submit your issue search the archive, maybe your question was already answered.

If your issue appears to be a bug, and hasn't been reported, open a new issue. Help us to maximize
the effort we can spend fixing issues and adding new features, by not reporting duplicate issues.
Providing the following information will increase the chances of your issue being dealt with
quickly:

* **Overview of the Issue**
* **Forge Version(s)** - is it a regression?
* **Operating System** - is this a problem on all OSes?
* **Reproduce the Error** - provide an example piece of code that can replicate the issue on our
  end.
* **Related Issues** - has a similar issue been reported before?
* **Suggest a Fix** - if you can't fix the bug yourself, perhaps you can point to what might be
  causing the problem (line of code or commit)

### Submitting a Pull Request
Before you submit your pull request consider the following guidelines:

* Search [GitHub](https://github.com/arrayfire/forge/pulls) for an open or closed Pull Request
  that relates to your submission. You don't want to duplicate effort.
* Make your changes in a new git branch:

    ```shell
    git checkout -b my-fix-branch hotfixes-major-minor-patch
    ```
    If there is no hotfixes branch, branch off of devel.

    ```shell
    git checkout -b my-fix-branch devel
    ```
* Commit your changes using a descriptive commit message that follows our
  [commit message conventions](#commit). Adherence to the [commit message conventions](#commit) is required.

    ```shell
    git commit -a
    ```
  Note: the optional commit `-a` command line option will automatically "add" and "rm" edited files.

* Build your changes locally to ensure everything checks out.
* Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```

In GitHub, send a pull request to `forge`. If we suggest changes, then:

* Make the required updates.
* Re-build and check corresponding examples work as expected.
* Commit your changes to your branch (e.g. `my-fix-branch`).
* Push the changes to your GitHub repository (this will update your Pull Request).

If the PR gets too outdated we may ask you to rebase and force push to update the PR:

_WARNING: Squashing or reverting commits and force-pushing thereafter may remove GitHub comments
on code that were previously made by you or others in your commits. Avoid any form of rebasing
unless necessary._

That's it! Thank you for your contribution!

#### After your pull request is merged

After your pull request is merged, you can safely delete your branch and pull the changes
from the upstream repository:

* Delete the remote branch on GitHub either through the GitHub web UI or your local shell
* Check out the parent(hotfixes-\*.\*.\*/devel) branch
* Delete the local branch
* Update the parent branch with the latest upstream version

## <a name="commit"></a> Git Commit Guidelines

We have very precise rules over how our git commit messages can be formatted.  This leads to **more
readable messages** that are easy to follow when looking through the **project history** and while
generating change log.

### Commit Message Format
Each commit message consists of a **header**, a **body** and a **footer**.  The header has a special
format that includes a **type**, a **scope** and a **subject**:

```
<type>: <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **header** is mandatory.

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read on GitHub as well as in various git tools.

### Revert
If the commit reverts a previous commit, it should begin with `revert: `, followed by the header of the reverted commit.
In the body it should say: `This reverts commit <hash>.`, where the hash is the SHA of the commit being reverted.

### Type
Must be one of the following:

* **feat**: A new feature
* **fix**: A bug fix
* **docs**: Documentation only changes
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing
  semi-colons, etc)
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **perf**: A code change that improves performance
* **misc**: Changes to the build process or auxiliary tools and libraries such as documentation
  generation

### Subject
The subject contains succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize first letter
* no dot (.) at the end

### Body
Just as in the **subject**, use the imperative, present tense: "change" not "changed" nor "changes".
The body should include the motivation for the change and contrast this with previous behavior.

### Footer
The footer should contain any information about **Breaking Changes** and is also the place to
[reference GitHub issues that this commit closes][closing-issues].

**Breaking Changes** should start with the word `BREAKING CHANGE:` with a space or two newlines.
The rest of the commit message is then used for this.

[github]: https://github.com/arrayfire/forge
[gitter]: https://gitter.im/arrayfire/forge
[groups]: https://groups.google.com/forum/#!forum/arrayfire-users
