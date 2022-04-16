# CI Failing Builds Detection

Continuous Integration is a set of modern Software Development practices widely adopted in commercial and open source environments that aim at supporting developers in integrating code changes constantly and quickly through an automated building and testing process.
This enables them to discover newly induced errors before they reach the end-user.

However, this build process is typically time and resource-consuming, specially for large projects with many dependencies where the build process can take hours or even days. This may cause disruptions in the development process and delays in the product release dates.

One of the main causes of these delays is that some commits needlessly start the Continuous Integration process, therefore ignoring them will significantly lower the cost of Continuous Integration without sacrificing its benefits.
Thus, withing the framework of this project, we aim to reduce the number of builds executed by skipping those deemed unnecessary.
