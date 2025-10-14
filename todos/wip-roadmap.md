# AI and Machine Learning Course Roadmap

---

## **CONTENTS**

### **Unit 1: Foundations of Artificial Intelligence and Business Impact**
* **Objective:** Establish a solid conceptual foundation, understand the "why" of AI and its implications.
* **Contents:**
    * Principles of intelligent systems: What is AI?
    * Strong AI vs. Weak AI: Characterization, uses, and possibilities.
    * AI vs Machine Learning vs Deep Learning: Definitions and relationships.
    * Data Science and Big Data: The role of data in AI.
    * AI lifecycle: From data collection to model deployment and monitoring.
    * Supervised, Unsupervised, and Reinforcement Learning: Basic concepts and differences (high-level overview).
    * AI application fields: Identification of sectors and use cases.
    * Technological Convergence: How AI, IoT, Cloud, and Blockchain unify processes and improve strategic decision-making.
    * AI in business: Operational efficiency improvement, new interactions, and business models.

---

### **Unit 2: Programming Ecosystem for AI**
* **Objective:** Acquire essential programming tools for AI development.
* **Contents:**
    * Introduction to Python: The industry standard language for AI.
    * Other languages: R, Julia, and their specific applications.
    * Environments with uv: Managing dependencies and virtual environments.
    * Development environments: Jupyter Notebooks, Google Colab, VS Code.
    * Setup: VSCode with Python, Jupyter, and Git. Environments and dependency management with uv.
    * **Programming with AI Agents:**
        * **GitHub Copilot in VS Code:** AI-powered code completion and generation.
        * Modern development workflow: Using AI as a pair programming partner.
        * Best practices: Prompt engineering for code generation, reviewing AI suggestions.
        * AI-assisted debugging and refactoring.
        * Leveraging AI agents throughout the development lifecycle.
        * **Note:** All course projects will be developed using modern AI-assisted workflows.
    * Version control with Git and GitHub: Collaboration and project management.
    * Fundamental libraries:
        * **NumPy:** For numerical computation.
        * **Pandas:** For data manipulation and analysis.

---


### **Project 1: Practice: Create a GitHub repository with a tic-tac-toe game. The machine should be able to play against a human using a mini-max algorithm.**
    - Improvements:
        - Implement a graphical user interface (GUI) using libraries like Tkinter or Pygame.
        - Add an option for difficult levels: agent plays randomly, agent uses minimax algorithm.
        - Add support for different difficulty levels by limiting the depth of the minimax search.
        - Add a scoring system to track wins, losses, and draws.
        - Include unit tests to ensure the correctness of the game logic and AI decisions.
--

### **Project 2: Processing MNIST with NumPy**

---

### **Unit 3: The Three Machine Learning Paradigms - A Practical Introduction**
* **Objective:** Understand the three fundamental approaches to machine learning through hands-on examples and prepare students for deeper dives into each paradigm.
* **Duration:** 1-2 weeks
* **Contents:**
    * **Overview: The Three Learning Paradigms**
        * What problems does each paradigm solve?
        * When to use supervised vs unsupervised vs reinforcement learning?
        * Real-world applications of each approach.
        * The learning signals: labels, patterns, and rewards.
    * **Paradigm 1: Supervised Learning - Learning from Examples**
        * Concept: Learning a mapping from inputs to outputs using labeled data.
        * **Simple example:** K-Nearest Neighbors (KNN) on Iris dataset.
        * Hands-on: Load pre-trained scikit-learn model, make predictions, evaluate accuracy.
        * Key insight: The model learns from examples with known answers.
        * Teaser: "In Unit 4, we'll dive deep into many supervised algorithms."
    * **Paradigm 2: Unsupervised Learning - Finding Hidden Patterns**
        * Concept: Discovering structure in data without labels.
        * **Simple example:** K-means clustering on 2D synthetic data.
        * Hands-on: Generate data, apply clustering, visualize results.
        * Key insight: The model groups similar data points automatically.
        * Teaser: "In Unit 5, we'll explore PCA, DBSCAN, and anomaly detection."
    * **Paradigm 3: Reinforcement Learning - Learning through Trial and Error**
        * Concept: An agent learns by interacting with an environment and receiving rewards.
        * Core components:
            * **Agent:** The learner/decision maker.
            * **Environment:** The world the agent interacts with.
            * **State:** Current situation of the agent.
            * **Action:** Choices available to the agent.
            * **Reward:** Feedback signal (positive or negative).
        * **Q-Learning Algorithm:**
            * Q-table: Storing expected rewards for state-action pairs.
            * Exploration vs Exploitation: Balancing trying new actions vs using known good ones.
            * Learning process: Updating Q-values based on experience.
            * Simple implementation with dictionaries and loops.
        * **Practical examples:**
            * GridWorld: Navigate a grid to reach a goal while avoiding obstacles.
            * Tic-Tac-Toe: Learn optimal moves through self-play.
        * Key insight: No one tells the agent what to do; it learns from consequences.
    * **Comparing the Paradigms:**
        * Comparison table: Data requirements, feedback type, common applications.
        * Discussion: Why different problems need different approaches.
    * **Python Programming Practice:**
        * Object-oriented programming: Classes for agents, environments, games.
        * Data structures: Dictionaries for Q-tables, lists for state representation.
        * Control flow: Loops for training episodes, conditionals for decision-making.
        * Code organization: Separating game logic, agent logic, and training code.
        * Testing: Unit tests for game rules and agent behavior.
    * **GitHub Workflow:**
        * Branching strategies for feature development.
        * Meaningful commit messages and version history.
        * Code documentation and README files.
        * Collaborative coding practices.

---

### **Project 3: Reinforcement Learning Game Agent**
*Applied after Unit 3 - ML Paradigms*
* **Objective:** Implement a Q-learning agent to master a simple game, practicing Python OOP and GitHub workflow.
* **Key Components:**
    * **Game environment:** Choose one of three difficulty levels:
        * **Level 1 (Beginner):** GridWorld navigation (5×5 grid, obstacles, goal).
        * **Level 2 (Intermediate):** Tic-Tac-Toe against random opponent.
        * **Level 3 (Advanced):** Tic-Tac-Toe against Minimax algorithm opponent.
    * **Q-learning implementation:**
        * State representation: How to encode game situations.
        * Action space: Valid moves from each state.
        * Reward design: +1 for win, -1 for loss, -0.1 for each move, etc.
        * Q-table update: Implementing the Q-learning formula.
        * Hyperparameters: Learning rate (α), discount factor (γ), exploration rate (ε).
    * **Training process:**
        * Training loop: Playing many episodes to improve.
        * Epsilon decay: Gradually reducing exploration over time.
        * Performance tracking: Win rate, average reward, training curves.
    * **Evaluation:**
        * Test against random opponent.
        * Test against rule-based strategies.
        * Visualize learned Q-values.
    * **Code quality requirements:**
        * Proper class structure (Agent, Environment, Game classes).
        * Documentation: Docstrings, comments, README.
        * Unit tests: Test game rules, move validation, Q-table updates.
        * Git workflow: Multiple commits with clear messages, branching if needed.
* **Learning outcomes:**
    * Understand RL concepts through implementation.
    * Practice Python programming and OOP design.
    * Develop professional coding habits (testing, documentation, version control).
    * Build confidence before tackling complex supervised learning algorithms.
* **Note:** This is a simplified, educational RL project focused on fundamentals and coding practice, not production-level RL systems.

---

### **Unit 4: Supervised Machine Learning - Deep Dive**
* **Objective:** Master the algorithms and techniques for learning from labeled data to make accurate predictions.
* **Contents:**
    * Machine Learning principles: Data, patterns, and predictions.
    * EDA and Data preprocessing: Cleaning, normalization, and feature selection.
    * **Regression:** Prediction of continuous values.
    * **Classification:** Prediction of categories.
    * Key Algorithms:
        * K-Nearest Neighbors (KNN)
        * Linear Regression
        * Logistic Regression
        * Decision Trees
        * Support Vector Machines (SVM)
        * Ensemble methods: Bagging, Boosting, and Stacking.
        * Random Forests
    * Using **Scikit-Learn** to implement these models.
    * Model evaluation: 
        * **For regression:** MAE, MSE, RMSE, R-squared, and residual analysis.
        * **For classification:** Accuracy, Precision, Recall, F1-Score, confusion matrix, ROC-AUC.
        * **General techniques:** Cross-validation and train/validation/test splits.
    * Optimization and hyperparameter tuning.

---


### **Project 4: Real Estate Price Prediction System on Miami Housing Dataset**
*Applied after Unit 4 - Supervised Learning*
* **Objective:** Complete regression project using real estate data to predict property prices.
* **Key Components:**
    * **Data acquisition:** Real estate datasets with multiple features (location, size, amenities, etc.).
    * **Feature engineering:** Creating new variables, handling categorical data, geographic features.
    * **Model comparison:** Linear Regression, Random Forest, XGBoost, SVM regression.
    * **Advanced evaluation:** Cross-validation, learning curves, residual analysis.
    * **Business application:** Price estimation tool with confidence intervals.
* **Neural Network Enhancement:** Implement MLP regression and compare with traditional ML models.


---

### **Unit 5: Unsupervised Machine Learning - Deep Dive**
* **Objective:** Master techniques for discovering hidden patterns and structures in unlabeled data.
* **Contents:**
    * **Clustering:** Grouping similar data points.
        * K-means, DBSCAN, and hierarchical clustering algorithms.
    * **Dimensionality Reduction:** Simplifying complex data.
        * **Principal Component Analysis (PCA):** Feature extraction and visualization.
        * t-SNE for high-dimensional data visualization.
        * Applications in feature engineering and noise reduction.
    * **Anomaly Detection:** Identification of outliers and unusual patterns.
    * Evaluation metrics for unsupervised learning.

---



### **Project 5: Intrusion Detection System (IDS) with Hybrid ML Approach**
*Applied after Unit 5 - Unsupervised Learning*
* **Objective:** Comprehensive cybersecurity project using both supervised and unsupervised techniques.
* **Key Components:**
    * **Real datasets:** NSL-KDD or CICIDS2017 for real network attacks.
    * **Phase 1 - Unsupervised:** 
        * PCA for dimensionality reduction and feature visualization.
        * Clustering/anomaly detection to establish normal behavior baseline.
        * DBSCAN and Isolation Forest for anomaly detection.
    * **Phase 2 - Supervised:** Classification of known attacks with high precision.
    * **Ensemble approach:** Combining both approaches to maximize detection and minimize false positives.
    * **Specialized metrics:** Balanced Precision/Recall to minimize false positives.
    * **Practical application:** Real-time dashboard for network monitoring.
* **Neural Network Enhancement:** Deep autoencoders for anomaly detection, CNNs for network traffic pattern recognition.

---

## 1ST TERM EVALUATION POINT

---

### **Unit 6: Deep Learning Fundamentals with PyTorch**
* **Objective:** Master the foundations of deep learning and neural network implementation from scratch using PyTorch.
* **Contents:**
    * **Neural Network Foundations:**
        * From Perceptron to Multi-Layer Perceptrons (MLP).
        * Universal approximation theorem and network depth.
        * Activation functions: ReLU, sigmoid, tanh, and modern variants.
        * The backpropagation algorithm: understanding gradient flow.
    * **Deep Learning with PyTorch:**
        * Introduction to **PyTorch** framework and tensor operations.
        * Building neural networks with `nn.Module`.
        * Training process: Loss functions, optimizers (SGD, Adam), and backpropagation.
        * Regularization techniques: Dropout, batch normalization, weight decay.
        * Training best practices: learning rate scheduling, early stopping.
        * Debugging neural networks: gradient checking, learning curves.
    * **Hands-on Implementation:**
        * Building MLPs from scratch for classification and regression.
        * MNIST digit classification with feedforward networks.
        * Hyperparameter tuning and model evaluation.
        * Visualizing training dynamics and decision boundaries.

---

### **Project 6: Neural Network Fundamentals - TensorFlow Playground Tutorial**
*Applied after Unit 6 - Deep Learning Fundamentals*
* **Objective:** Create a comprehensive tutorial covering all scenarios in TensorFlow Playground to build intuition about neural network behavior.
* **Key Components:**
    * **Systematic exploration:** Document behavior across all datasets and architectures.
    * **Network depth analysis:** Compare shallow vs deep networks.
    * **Activation functions:** Experiment with different activation functions.
    * **Learning rate effects:** Demonstrate impact on convergence.
    * **Regularization:** Show effects of dropout and other techniques.
    * **Overfitting visualization:** Demonstrate and explain overfitting patterns.
* **Learning outcomes:** Build strong intuition before implementing networks from scratch.

---

### **Unit 7: Computer Vision with Convolutional Neural Networks**
* **Objective:** Master CNN architectures and modern computer vision techniques using pre-trained models and transfer learning.
* **Contents:**
    * **Convolutional Neural Networks (CNNs) - Deep Dive:**
        * The challenge of image data: spatial structure and parameter efficiency.
        * CNN architecture components:
            * Convolutional layers: filters, feature maps, receptive fields.
            * Pooling layers: max pooling, average pooling.
            * Fully connected layers and feature extraction.
        * Understanding convolutions: visualizing filters and feature maps.
        * Classic architectures: LeNet, AlexNet, VGG, ResNet, EfficientNet.
    * **Transfer Learning and Pre-trained Models:**
        * Why transfer learning works: learned features and domain adaptation.
        * Using pre-trained models from torchvision and timm.
        * Fine-tuning strategies: freezing layers, learning rates.
        * Feature extraction for downstream tasks.
    * **Modern Computer Vision Applications:**
        * **Object Detection:** YOLO, Faster R-CNN architectures.
        * **Semantic Segmentation:** U-Net and FCN concepts.
        * **Image Similarity:** Feature extraction and similarity search.
        * **Real-time inference:** Optimization for production.
    * **Autoencoders for Unsupervised Learning:**
        * Architecture: encoder-decoder structure and bottleneck.
        * Latent space representation and dimensionality reduction.
        * Training objectives: reconstruction loss.
        * Applications: denoising, compression, feature learning, anomaly detection.
        * Variational Autoencoders (VAE): probabilistic approach to generative modeling.
        * Comparison with PCA and other dimensionality reduction techniques.
    * **Practical Tools and Frameworks:**
        * torchvision: datasets, models, transforms.
        * Hugging Face transformers for vision models.
        * YOLO frameworks: ultralytics, YOLOv8.
* **Available Examples:**
    * **MNIST Examples:**
        * `FNN_MNIST.ipynb` - Multi-layer perceptron with normalization and regularization techniques
        * `CNN_MNIST.ipynb` - Basic CNN architecture for digit classification
        * `mnist-model-comparison.py` - Comparative analysis between logistic regression, single-layer NN, and CNNs
    * **CIFAR-10 Examples:**
        * `CIFAR-10.ipynb` - Complete CIFAR-10 classification pipeline
        * `CIFAR10_gray.ipynb` - FNN vs CNN comparison on grayscale CIFAR-10
        * `models/baseline.py` and `models/mejorado.py` - Basic and improved CNN architectures
    * **Key Learning Concepts:**
        * Data preprocessing and normalization techniques
        * Architecture design principles for FNN vs CNN
        * Training loop implementation and optimization
        * Performance comparison and error analysis
        * Understanding why CNNs excel at image tasks
* **Learning outcomes:** Students will understand neural network fundamentals, see practical implementations, and comprehend the advantages of CNNs for computer vision tasks.

---

### **Project 7B: Intelligent Computer Vision System**
*Applied after Unit 7 - Computer Vision with CNNs*
* **Objective:** Complete computer vision pipeline using pre-trained models, object detection, and similarity search with high-level libraries.
* **Key Components:**
    * **YOLO implementation:** Real-time object detection in images/video using ultralytics or similar.
    * **Feature extraction:** Using pre-trained CNNs (ResNet, EfficientNet) to extract image features.
    * **Vector database:** Storing image features in ChromaDB for similarity search.
    * **Similarity search:** Finding visually similar images using vector embeddings.
    * **Transfer learning:** Fine-tune a model on a custom dataset.
    * **Practical application:** Visual search engine for e-commerce or content management.
* **Advanced enhancements:** Integration with LLMs for image description and automated tagging.
* **Learning outcomes:** Apply modern computer vision tools to solve real-world problems using high-level APIs.
* **Note:** This project focuses on using production-ready tools, not implementing algorithms from scratch.

---

### **Unit 8: Sequential Data and the Transformer Revolution**
* **Objective:** Master specialized architectures for sequential and temporal data, from RNNs to modern Transformers.
* **Contents:**
    * **The Challenge of Sequential Data:**
        * Characteristics of sequences: variable length, temporal dependencies, context.
        * Applications: Natural Language Processing (NLP), time series, speech, video.
        * Why traditional neural networks struggle with sequences.
    * **Recurrent Neural Networks (RNNs):**
        * Architecture: hidden state as memory mechanism.
        * Processing sequences: one element at a time.
        * Training RNNs: Backpropagation Through Time (BPTT).
        * **RNN variants:**
            * LSTM (Long Short-Term Memory): gates for selective memory.
            * GRU (Gated Recurrent Units): simplified gating mechanism.
            * Bidirectional RNNs: context from both directions.
        * **Limitations:** Vanishing gradients, sequential processing bottleneck, difficulty with long-range dependencies.
    * **Sequence-to-Sequence (Seq2seq) Models:**
        * Encoder-decoder architecture for variable-length input/output.
        * Applications: machine translation, text summarization, chatbots.
        * Limitations: information bottleneck, fixed context vector.
    * **The Attention Mechanism - The Breakthrough:**
        * Motivation: allowing the model to focus on relevant parts of the input.
        * **Attention fundamentals:**
            * Query, Key, Value paradigm.
            * Attention scores and weighted sum.
            * Soft attention vs hard attention.
        * Attention in seq2seq: dynamic context for each decoding step.
    * **Transformer Architecture - "Attention is All You Need":**
        * Eliminating recurrence: parallel processing of sequences.
        * **Self-Attention mechanism:**
            * Relating different positions within a single sequence.
            * Multi-head attention: capturing diverse relationships in parallel.
            * Complexity and efficiency considerations.
        * **Positional encodings:** Injecting sequence order information.
        * **Transformer building blocks:**
            * **Encoder architecture:**
                * Self-attention for bidirectional context understanding.
                * Feed-forward networks and residual connections.
                * Layer normalization.
                * Applications: BERT, RoBERTa for classification and understanding.
            * **Decoder architecture:**
                * Masked self-attention for autoregressive generation.
                * Prevents looking at future tokens during training.
                * Applications: GPT family for text generation.
            * **Encoder-Decoder architecture:**
                * Cross-attention: connecting encoder outputs to decoder.
                * Applications: T5, BART for translation, summarization, seq2seq tasks.
        * **Transformers vs RNNs:** Advantages in parallelization, long-range dependencies, and efficiency.
    * **Hugging Face Ecosystem - Practical Transformers:**
        * Introduction to **Hugging Face** libraries:
            * `transformers`: Pre-trained models for various tasks.
            * `datasets`: Access to thousands of datasets.
            * `tokenizers`: Fast text processing and preparation.
        * Working with pre-trained models:
            * Model selection: BERT, GPT, T5, RoBERTa, and specialized variants.
            * Tokenization and input preparation.
            * Inference for common tasks: classification, generation, question answering.
        * **Fine-tuning:** Adapting pre-trained models to specific tasks.
            * Transfer learning in NLP.
            * Training strategies and hyperparameters.
            * Efficient fine-tuning: LoRA and parameter-efficient methods.
    * **Transformers Beyond NLP:**
        * **Time series forecasting:** Temporal attention for predictions.
        * Vision Transformers (ViT): applying transformers to images.
        * Multi-modal transformers: combining text, images, and other modalities.

---

### **Project 8: Evolution of NLP Architectures - From Traditional ML to Transformers**
*Applied after Unit 8 - Sequential Data and Transformers*
* **Objective:** Comprehensive comparison of NLP architectures through the evolution of text classification approaches.
* **Key Components:**
    * **Consistent Problem:** Fake news detection throughout all phases for direct comparison.
    * **Phase 1 - Traditional ML:** TF-IDF features with SVM, Naive Bayes, and ensemble methods.
    * **Phase 2 - Neural Networks:** Multi-layer perceptrons with word embeddings.
    * **Phase 3 - CNNs:** Convolutional networks for text classification with different filter sizes.
    * **Phase 4 - RNNs:** LSTM and bidirectional LSTM networks for sequential processing.
    * **Phase 5 - Transformers:** Fine-tuned BERT models for state-of-the-art performance.
    * **Advanced Analysis:** Performance comparison, computational cost analysis, and interpretability study.
    * **Practical Application:** Deployment-ready fake news detection API with multiple model options.
* **Learning outcomes:** Understand the evolution of NLP and when to use each architecture.
* **Note:** Uses high-level libraries (scikit-learn, PyTorch, Hugging Face) to focus on architecture comparison.

---

### **Unit 9: Generative AI and Large Language Models (LLMs)**
* **Objective:** Understand and utilize models capable of generating new and coherent content.
* **Contents:**
    * **Generative Models Landscape:**
        * What is Generative AI? Overview of different approaches.
        * **Generative Adversarial Networks (GANs):**
            * Generator vs Discriminator: adversarial training concept.
            * Applications: image generation, data augmentation, style transfer.
            * Training challenges: mode collapse, convergence issues.
            * Historical importance and current use cases.
        * **Autoregressive Models:**
            * Sequential generation: predicting next token based on previous ones.
            * Architecture examples: GPT family, traditional language models.
            * Applications: text generation, code completion, time series forecasting.
            * Strengths: coherent long-form generation, flexibility.
        * **Diffusion Models:**
            * Forward process: gradually adding noise to data.
            * Reverse process: learning to denoise and generate samples.
            * Modern applications: DALL-E 2, Midjourney, Stable Diffusion.
            * Advantages over GANs: training stability, sample quality, mode coverage.
        * **Comparison of generative approaches:** When to use each paradigm.
    * **Large Language Models (LLMs):**
        * Architecture foundation: Transformer-based autoregressive models (GPT).
        * Scale and emergence: how size enables new capabilities.
        * Pre-training and fine-tuning paradigm.
    * **Prompt Engineering:** The art of designing effective inputs for LLMs.
    * **LLM application development:**
        * **Direct approach**: Native APIs (OpenAI, Anthropic) for maximum control and transparency.
        * **LLM orchestration frameworks:**
            * **LangChain**: General-purpose framework for building LLM applications.
                * Chains: Composing LLM calls and logic.
                * Agents: Dynamic tool selection and reasoning.
                * Memory: Conversation history and context management.
                * Document loaders and text splitters.
            * **LlamaIndex**: Specialized framework for RAG and knowledge retrieval.
                * Index construction and query engines.
                * Advanced retrieval strategies.
                * Integration with vector stores.
            * **Comparison**: When to use each framework and how they complement each other.
        * Memory and context management in conversational applications.
    * **Vector databases and embeddings:**
        * Concepts: similarity search, high-dimensional embeddings.
        * **ChromaDB**: Practical and easy-to-use vector database.
        * **Practical application**: Implementation of semantic search and RAG with real documents.
    * **RAG (Retrieval-Augmented Generation):** Augmenting LLM knowledge with external documents to reduce hallucinations and use private data.
    * **Agents and Model Context Protocol:**
        * Creating autonomous systems that use LLMs to reason and execute actions.
        * **Model Context Protocol (MCP):** Standard for connecting LLMs with external tools and data.
        * Implementation of agents with advanced interaction capabilities.

---

### **Project 9: Intelligent Document Processing with RAG**
*Applied after Unit 9 - Generative AI*
* **Objective:** Complete RAG system for document analysis and question-answering.
* **Key Components:**
    * **Document ingestion:** PDF, Word, web scraping for diverse document sources.
    * **Text processing:** Chunking strategies, embedding generation with different models.
    * **Vector storage:** ChromaDB implementation with metadata filtering.
    * **RAG pipeline:** Query processing, retrieval, and generation with LLMs.
    * **Evaluation:** Retrieval accuracy, answer quality, and hallucination detection.
    * **Practical application:** Corporate knowledge base or legal document analysis system.
* **Advanced features:** Multi-modal RAG with images, conversation memory, and agent integration.


---

### **Unit 10: MLOps and Production Machine Learning**
* **Objective:** Master the practices and tools for deploying, monitoring, and maintaining ML models in production environments.
* **Motivation:** Bridging the gap between model development and real-world deployment - the most critical skill for ML practitioners.
* **Contents:**
    * **The MLOps Landscape:**
        * What is MLOps? DevOps principles applied to ML.
        * The ML production lifecycle: development, deployment, monitoring, retraining.
        * Common challenges: model drift, data drift, reproducibility, scaling.
        * MLOps maturity levels: from manual to fully automated.
    * **Experiment Tracking and Model Management:**
        * **MLflow:** Tracking experiments, parameters, metrics, and artifacts.
        * **Weights & Biases (W&B):** Advanced experiment tracking and visualization.
        * Model registry: Versioning models and managing their lifecycle.
        * Comparing experiments: Finding the best model systematically.
        * Reproducibility: Ensuring experiments can be replicated.
    * **Model Deployment Strategies:**
        * **Containerization with Docker:**
            * Creating Docker images for ML models.
            * Dockerfile best practices for ML applications.
            * Managing dependencies and environment consistency.
        * **API Development:**
            * **FastAPI:** Creating RESTful endpoints for model inference.
            * Request validation, error handling, and documentation.
            * Synchronous vs asynchronous inference.
            * Batch prediction endpoints.
        * **Deployment patterns:**
            * Blue-green deployment for zero downtime.
            * Canary releases for gradual rollout.
            * A/B testing for model comparison.
            * Shadow mode for safe testing.
    * **Cloud Deployment:**
        * Overview of cloud ML platforms:
            * **AWS SageMaker:** End-to-end ML platform.
            * **Azure Machine Learning:** Microsoft's ML suite.
            * **Google Cloud Vertex AI:** Unified ML platform.
        * Serverless deployment: AWS Lambda, Azure Functions, Google Cloud Functions.
        * Managed endpoints and auto-scaling.
        * Cost optimization strategies.
    * **Model Monitoring and Observability:**
        * **Performance monitoring:** Tracking latency, throughput, error rates.
        * **Model drift detection:**
            * Data drift: Changes in input distributions.
            * Concept drift: Changes in the relationship between inputs and outputs.
            * Tools: Evidently AI, NannyML, Alibi Detect.
        * **Alerting systems:** Setting up alerts for degraded performance.
        * **Logging best practices:** Structured logging for ML systems.
        * **Metrics dashboards:** Grafana, Prometheus for ML monitoring.
    * **CI/CD for Machine Learning:**
        * Continuous Integration: Automated testing for ML code and data.
        * Continuous Deployment: Automated model deployment pipelines.
        * **GitHub Actions:** Setting up ML pipelines.
        * Testing ML systems:
            * Unit tests for data preprocessing and feature engineering.
            * Integration tests for model serving.
            * Smoke tests for deployed models.
        * **Data validation:** Great Expectations for data quality checks.
    * **Model Versioning and Governance:**
        * **Data Version Control (DVC):** Versioning datasets and models.
        * Model lineage: Tracking model origins and transformations.
        * Model documentation and metadata.
        * Compliance and audit trails.
    * **Scaling ML Systems:**
        * Horizontal vs vertical scaling.
        * Load balancing for model inference.
        * Caching strategies for common predictions.
        * Batch vs real-time inference trade-offs.
        * **Kubernetes for ML:** Basics of container orchestration.
    * **Advanced Topics:**
        * **Edge deployment:** TensorFlow Lite, ONNX for mobile and IoT.
        * **Model optimization:** Quantization, pruning, distillation.
        * **Multi-model serving:** Hosting multiple models efficiently.
        * **Feature stores:** Centralized feature management (Feast, Tecton).


---

## Transversal topics (to be taken into account but not as a separate unit)

### Ethics and Interpretability in AI
    * **Bias and Fairness:**
        * Sources of bias: Data bias, algorithmic bias, interaction bias.
        * **Types of fairness:**
            * Individual fairness vs group fairness.
            * Demographic parity, equalized odds, calibration.
        * **Bias detection:** Statistical tests and fairness metrics.
        * **Bias mitigation:**
            * Pre-processing: Reweighting, sampling strategies.
            * In-processing: Fair learning algorithms.
            * Post-processing: Threshold adjustment.
        * **Tools:** AI Fairness 360, Fairlearn.
    * **Model Interpretability and Explainability:**
        * **Why interpretability matters:** Trust, debugging, compliance, scientific discovery.
        * **Global vs local interpretability:**
            * Global: Understanding the model's overall behavior.
            * Local: Explaining individual predictions.
        * **Intrinsically interpretable models:**
            * Decision trees, linear models, rule-based systems.
            * Trade-offs: Interpretability vs performance.
        * **Post-hoc explanation methods:**
            * **SHAP (SHapley Additive exPlanations):** Unified framework for model explanations.
            * **LIME (Local Interpretable Model-agnostic Explanations):** Local surrogate models.
            * **Attention visualization:** Understanding what transformers focus on.
            * **Saliency maps:** Visualizing important regions in images.
        * **Practical tools:** SHAP, LIME, Captum (for PyTorch), InterpretML.
   

---

## **Optional Advanced Topics** (Time Permitting or Self-Study)

### **Time Series Forecasting**
* ARIMA, SARIMA, and classical methods.
* Prophet for business time series.
* Transformer-based forecasting (Temporal Fusion Transformer).
* Applications: Demand forecasting, financial predictions, anomaly detection in time series.

### **Multimodal AI**
* CLIP: Connecting vision and language.
* Vision-language models for image captioning and VQA.
* Audio processing and speech recognition.
* Multimodal embeddings and applications.

### **Advanced Deep Learning Topics**
* Graph Neural Networks (GNNs) for graph-structured data.
* Meta-learning and few-shot learning.
* Self-supervised learning techniques.
* Neural Architecture Search (NAS).

### **Advanced LLM Techniques**
* **Fine-tuning at scale:** LoRA, QLoRA, PEFT in depth.
* **Prompt engineering:** Advanced techniques, chain-of-thought, few-shot learning.
* **LLM evaluation:** Benchmarks, human evaluation, automated metrics.
* **Constitutional AI and RLHF:** Aligning models with human values.
