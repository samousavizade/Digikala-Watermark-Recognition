# Dependencies

- OS `os`
- MLFlow `mlflow`
- PyTorch, PyTorch Vision `torch`
- Optuna `optuna`

---

# Project Definition

In the **Digikala** sellers panel, when sellers add their products to the website for
sale, they must also submit a number of images of that product for each product; But not
every image can be appropriate because images must meet a predetermined conditions.
One of these conditions is the absence of
any watermarks on the image. In this issue,
we ask you to use the data provided to you to train a model that is able to detect
the presence of watermarks.

One of these conditions is the **absence of any watermarks** on the
image. In this case, we want to use the data we have to teach a model
that is able to detect the presence of watermarks.

The data set we have has **training** and **test** parts. In the training part,
a set of images that have a watermark are in a
`positive` folder and a set of images that
do not have any watermark are in a `negative` folder.

Using these images, we train our machine learning algorithm
and then predict whether each of the images in the test folder has a watermark.

---

# Learning Model

In this project, we use **Inception V3** **End-To-End** pre-trained deep neural network with the help of **Transfer Learning** technique.

Transfer learning brings a range of benefits to the development process of machine learning models. The main benefits of transfer learning include the saving of resources and improved efficiency when training new models.

> [End-to-End Models](https://www.capitalone.com/tech/machine-learning/pros-and-cons-of-end-to-end-models/)

> [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning#:~:text=Transfer%20learning%20(TL)%20is%20a,when%20trying%20to%20recognize%20trucks.)

End-to-end models have a number of advantages relative to component-based systems, but they also have some disadvantages.

Advantages of end-to-end models:
- **Better metrics**: Currently, the systems with the best performance according to metrics such as precision and recall tend to be end-to-end models.
- **Simplicity**: End-to-end models avoid the sometimes thorny problem of determining which components are needed to perform a task and how those components interact. In component-based systems, if the output format of one component is changed, the input format of other components may need to be revised.
- **Reduced effort**: End-to-end models arguably require less work to create than component-based systems. Component-based systems require a larger number of design choices.
- **Applicability to new tasks**: End-to-end models can potentially work for a new task simply by retraining using new data. Component-based systems may require significant re-engineering for new tasks.
- **Ability to leverage naturally-occurring data**: End-to-end models can be trained on existing data, such as translations of works from one language to another, or logs of customer service agent chats and actions. Component-based systems may require creation of new labeled data to train each component.
- **Optimization**: End-to-end models are optimized for the entire task. Optimization of a component-based system is difficult. Errors accumulate across components, with a mistake in one component affecting downstream components. Information from downstream components can’t inform upstream components.
- **Lower degree of dependency on subject matter experts**: End-to-end models can be trained on naturally-occurring data, which reduces the need for specialized linguistic and domain knowledge. But expertise in deep neural networks is often required.
- **Ability to fully leverage machine learning**: End-to-end models take the idea of machine learning to the limit.
