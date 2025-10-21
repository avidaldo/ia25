# Essential Tools for AI Programming

## Programming Languages

### Python

Python is the **predominant programming language for *machine learning***. It will be the language we use throughout this course. Python is an interpreted, high-level, general-purpose programming language. It is a very versatile and easy-to-learn programming language, and has a **vast collection of libraries and frameworks for AI**, *machine learning*, and *deep learning*.

Python programming courses:
- https://edube.org/study/pe1
- https://edube.org/study/pe2


### Other Relevant Programming Languages in AI

- R: Primarily used in statistics and data analysis.
- C/C++: Many of the libraries accessed from Python are implemented in C/C++ since they are low-level languages and much faster than Python. C++ remains the reference when performance is critical.
- Julia: High-performance programming language designed for scientific and technical computing.
- Mojo: Aims to achieve Python's usability with performance similar to C.


## Git and Git Forges

Git is a distributed version control system that allows us to keep a record of changes in a project's source code.
Git forges are services that allow us to host our Git repositories in the cloud. Some of the most popular forges are GitHub, GitLab, and Bitbucket. Throughout this course, we will extensively use Git and GitHub.

- [Git and GitHub Resources](https://github.com/avidaldo/recursos-git)
- [Git and GitHub for AI Programming](https://www.youtube.com/watch?v=T1tYBbiWTbc)


## IDEs (Integrated Development Environment)

For working with Python for *machine learning* and *deep learning*, some of the most popular IDEs are:

### [Visual Studio Code (VS Code)](https://code.visualstudio.com/)

Visual Studio Code is a source code editor developed by Microsoft. It is cross-platform and supports multiple programming languages. It is one of the most popular editors among software developers.

- [VISUAL STUDIO CODE: Beginner's Tutorial](https://www.youtube.com/watch?v=CxF3ykWP1H4)

### [Cursor](https://www.cursor.com/)

Cursor is a [fork](https://en.wikipedia.org/wiki/Fork_(software_development)) of VS Code focused on using LLMs for programming.

### [PyCharm](https://www.jetbrains.com/pycharm/)

PyCharm is an IDE developed by JetBrains. It is one of the most popular IDEs for Python and is used by many software developers.

### [DataSpell](https://www.jetbrains.com/dataspell/)

DataSpell is another JetBrains IDE similar to PyCharm but specialized in data analysis.

## Jupyter Notebooks

Jupyter is an application that allows us to create *notebooks* (documents that combine code, rich text, equations, visualizations, etc.) in the browser. These *notebooks* are files with the .ipynb extension (when using the Python language) that can be exported in different static formats, such as HTML or PDF. The `.ipynb` files are internally JSON.

## Virtual Environments

**Virtual environments** allow us to isolate a project's **dependencies** from those of the operating system. This enables us to have different versions of the same **libraries** in different projects without conflicts between them.

 - [Conda Tutorial: Environment and Package Management](conda_tutorial.md)
 - [uv Tutorial: A Fast Python Package Installer and Resolver](uv_tutorial.md)

 - [Introduction to pip, conda, and virtual environments](https://www.youtube.com/watch?v=7Rd-Gj8o-6Q)

We will mainly use the more recent `uv`; a modern tool for managing virtual environments and dependencies in Python projects.

 - [UV for Python… (Almost) All Batteries Included](https://www.youtube.com/watch?v=qh98qOND6MI)

### [Dependency and Virtual Environment Management in Python](py_dependencies.md)


## [Google Colab](https://colab.research.google.com/)

Google Colab is a free Google service that allows us to run Jupyter *notebooks* in the cloud. Colab allows us to execute code in Python, R, and other programming languages, and provides access to hardware resources such as GPUs and TPUs.

