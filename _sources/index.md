# Goals and To-Dos

## Goals
- Consolidate all of my AI/ML notes in one location where they will be searchable and accessible to me from anyhwere
- Gain a working understanding of a wide variety of AI/ML techniques, architectures, and concepts
- Build my AI Portfolio by using many types of AI

## To-Dos
- cover topics suggested to me
    - ML
        - high level overview of supervised/unsupervised/cluster evaluation
        - dimensionality reduction (methods on the decision tree)
        - review the main ML algorithms from the Scikit Learn docs
    - NLP/LLM  
        - Crash Course on NLP (I think this may be covered by Kaggle's "Intro to NLP" tutorial, though it could be good to go through their "Iterate like a Grandmaster" tutorial)
        - Transformers (watch the 3blue1brown video on these and take notes)
        - RAG
- cover topics mentioned on specific postings
    - SHAP, LIME (exlainable Ai)
    - SQL Query-writing

## Resources
Some of the resources available to me to learn about/practice AI.

### [Machine Learning Mastery](https://machinelearningmastery.com/)
Spenser strongly suggested this site
- mentions [Building an AI Portfolio](https://machinelearningmastery.com/build-a-machine-learning-portfolio/)
- suggests [7 LLM Projects to Boost Your Machine Learning Portfolio](https://machinelearningmastery.com/7-llm-projects-to-boost-your-machine-learning-portfolio/)
- has guides like [4 Types of Classification Tasks in Machine Learning](https://machinelearningmastery.com/types-of-classification-in-machine-learning/). Usually the guides very briefly discuss the theory and some network options, then show a code block and a graphic. I don't learn as well if I'm just copying and pasting, so I'll need to figure out how to transform the information in order to learn the concepts well

### [Kaggle](https://www.kaggle.com/) 
- Widely known for hosting public coding competitions and datasets, they have a lot of potential in guiding growth and getting real practice
- [Learn](https://www.kaggle.com/learn) section with courses like these. The tutorials don't rely on videos, and the Exercises launch Jupyter notebook environments which you appear to be able to use without an account, and let you download the data sets. [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning), [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning), [Feature Engineering](https://www.kaggle.com/learn/feature-engineering), [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning), [Natural Language Processing Guide](https://www.kaggle.com/learn-guide/natural-language-processing). 
- [Competitions](https://www.kaggle.com/competitions) also exists, and may be the optimal way for me to learn - hand me dataset and a challenge and force the actual decision-making process. Might be best to start on the [Getting Started](https://www.kaggle.com/competitions?hostSegmentIdFilter=5) or [Playground](https://www.kaggle.com/competitions?hostSegmentIdFilter=8) **Competitions**. You need to accept the rules for a competition before downloading the dataset, though you don't have to join it. Their [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) walks you through each step of that process.
- [API](https://github.com/Kaggle/kaggle-api) can be pip-installed in order to programatically download/upload **Datasets**, interface with **Competitions**/**Kernels**/**Models**, etc. (see [Documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md))

### [3Blue1Brown](https://www.3blue1brown.com/)
- Has a [YouTube Channel](https://www.youtube.com/c/3blue1brown) where he goes over a lot of complex topics, using advanced visuals to visualize advanced math.
- Created the [manim](https://github.com/3b1b/manim) Python Library to programmatically animate math (documentation [here](https://3b1b.github.io/manim/)).

### Emergent Garden
- [Watching Neural Networks Learn (Emergent Garden)](https://www.youtube.com/watch?v=TkwXa7Cvfr8&t=1105s)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) 
    - (“book” site recommended by Emergent Garden)
    - it has good examples, such as this one on [Digit Classification](http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits)
    - can download the examples or use `git clone https://github.com/mnielsen/neural-networks-and-deep-learning.git`

### [GeeksForGeeks](https://www.geeksforgeeks.org/artificial-intelligence/what-is-artificial-intelligence-ai/) 
Has a bunch of articles with and without code examples. I've linked a ton of them throughout my notes already.

### [Scikit-Learn Docs](https://scikit-learn.org/1.7/index.html)
- scikit-learn.org seems more focused on teaching AI/ML concepts than being a framework for powerful custom deployed ML methods. 
- the front page is broken into categories - click on any of them to drill down into guidance for ML fitting into those categories.

### [DataCamp](https://app.datacamp.com/) 
- Free for me through work, has [Career Tracks](https://app.datacamp.com/learn/career-tracks) like [Machine Learning Scientist in Python](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) which comprise multiple courses like [Supervised Learning with scikit-learn](https://app.datacamp.com/learn/courses/supervised-learning-with-scikit-learn), which has chapters on [Classification](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=1), [Regression](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-7f892f18-f9c3-4c6f-9570-f19ed117c967?ex=1), [Fine-Tuning your Model](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/fine-tuning-your-model-3?ex=1), and [Preprocessing and Pipelines](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=1). 
- Courses consist of videos followed by code exams. It's not my preferred style of learning, and you have to do a lot of extra work to extract the example code and use it yourself.

### Functional Sites
These sites have some useful functionality not directly related to AI/ML
- [30secondsofcode.org](https://30secondsofcode.org) - snippets
- [neumorphism.io](https://neumorphism.io) - CSS generator
- [crontab guru](https://crontab.guru) - cron schedule editor
- [learn git branching](https://learngitbranching.js.org) - git branching
- [regexr.com](https://regexr.com) - Regex live checking

```{toctree}
:caption: AI/ML:

categories/categories
fundamentals/fundamentals
libraries/libraries
```

```{toctree}
:caption: Sphinx:

sphinx/sphinx
```
