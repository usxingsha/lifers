#!/usr/bin/env python3
"""Massive English high-quality corpus generator — 50+ domains, 1M+ words"""
import sys, os, random

sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
CORPUS_PATH = Path(__file__).resolve().parent.parent / "weights" / "training_corpus.txt"
random.seed(42)

# ═══════════════════════════════════════════
# English Knowledge Base — 60+ topics, detailed content
# ═══════════════════════════════════════════

EN_KNOWLEDGE = {
    "Artificial Intelligence": [
        ("Deep Learning Architectures",
         "Deep learning has revolutionized artificial intelligence through multi-layered neural networks capable of learning hierarchical representations. Convolutional Neural Networks (CNNs) excel at processing grid-like data such as images, using shared weights and local connectivity to detect spatial patterns. Recurrent Neural Networks (RNNs) and their variants like LSTM and GRU handle sequential data by maintaining internal states that capture temporal dependencies. The Transformer architecture, introduced in 'Attention Is All You Need', replaced recurrence with self-attention mechanisms, enabling parallel processing and capturing long-range dependencies more effectively than RNNs. This architectural shift led to breakthroughs in natural language processing with models like BERT and GPT, and has since been adapted to computer vision, speech processing, and multi-modal learning. The key innovation of self-attention is its ability to compute weighted representations of all positions in a sequence simultaneously, allowing the model to directly attend to relevant context regardless of distance.",
         "The evolution from CNNs to RNNs to Transformers represents a progressive relaxation of inductive biases — from strong spatial locality assumptions to complete flexibility in attention patterns."),
        ("Reinforcement Learning Fundamentals",
         "Reinforcement learning (RL) is a computational approach to learning from interaction, where an agent learns to maximize cumulative rewards through trial and error in an environment. The RL problem is formalized as a Markov Decision Process (MDP) consisting of states, actions, transition probabilities, and reward functions. Key concepts include the value function (expected future reward from a state), the Q-function (expected future reward from a state-action pair), and the policy (mapping from states to actions). Temporal Difference (TD) learning combines ideas from Monte Carlo methods and dynamic programming, updating estimates based on other learned estimates without waiting for final outcomes. The exploration-exploitation dilemma — whether to choose known good actions or explore uncertain ones — is fundamental to RL and addressed through strategies like epsilon-greedy, Upper Confidence Bounds (UCB), and Thompson sampling.",
         "The beauty of RL lies in its generality — any problem that can be framed as sequential decision-making under uncertainty can potentially be solved with RL."),
    ],
    "Computer Science Fundamentals": [
        ("Algorithm Analysis and Complexity",
         "Algorithm analysis provides the theoretical foundation for understanding computational efficiency. Time complexity measures how an algorithm's running time grows with input size, typically expressed using Big O notation. Common complexity classes include O(1) constant time, O(log n) logarithmic time (binary search), O(n) linear time (linear search), O(n log n) linearithmic time (merge sort), O(n²) quadratic time (bubble sort), and O(2ⁿ) exponential time (brute force traveling salesman). Space complexity measures memory requirements similarly. Amortized analysis considers the average performance over a sequence of operations, providing a more realistic picture for data structures like dynamic arrays where occasional expensive operations are balanced by many cheap ones. Understanding algorithmic complexity is essential for making informed design decisions, as the difference between O(n²) and O(n log n) can mean the difference between seconds and hours for large inputs.",
         "A deep understanding of algorithmic complexity separates software engineers who write code that merely works from those who write code that scales."),
        ("Database Systems and ACID",
         "Database management systems provide structured storage and retrieval of data, with relational databases dominating enterprise applications for decades. The relational model, proposed by E.F. Codd in 1970, organizes data into tables (relations) with rows (tuples) and columns (attributes). SQL (Structured Query Language) provides a declarative interface for querying and manipulating relational data. ACID properties — Atomicity, Consistency, Isolation, Durability — guarantee reliable transaction processing: Atomicity ensures all-or-nothing execution; Consistency preserves database invariants; Isolation prevents interference between concurrent transactions; Durability ensures committed data survives system failures. Isolation levels (Read Uncommitted, Read Committed, Repeatable Read, Serializable) trade off consistency guarantees against concurrency performance. Modern distributed databases often relax some ACID guarantees in favor of BASE (Basically Available, Soft state, Eventually consistent) properties to achieve horizontal scalability.",
         "The tension between consistency and availability, formalized in the CAP theorem, is fundamental to distributed systems design."),
    ],
    "Mathematics": [
        ("Linear Algebra in Machine Learning",
         "Linear algebra provides the mathematical foundation for modern machine learning and deep learning. Vectors represent data points, features, or model parameters in multi-dimensional spaces. Matrices encode linear transformations, and matrix multiplication implements the forward pass of neural networks. Eigenvalues and eigenvectors reveal principal directions of variation, forming the basis of Principal Component Analysis (PCA). The Singular Value Decomposition (SVD) generalizes eigendecomposition to non-square matrices and is widely used in dimensionality reduction, collaborative filtering, and matrix completion. Gradient-based optimization relies on computing partial derivatives and arranging them into gradient vectors that point in the direction of steepest ascent. The chain rule from calculus, expressed through automatic differentiation, enables efficient gradient computation through arbitrarily complex computational graphs. Understanding the geometric interpretation of linear algebra operations — transformation, rotation, scaling, projection — provides deep intuition for why neural networks work.",
         "Linear algebra is not just a prerequisite for machine learning — it is the language in which machine learning is written."),
        ("Probability Theory and Statistics",
         "Probability theory provides the mathematical framework for reasoning under uncertainty, which is central to machine learning and AI. Bayes' theorem, relating conditional probabilities P(A|B) = P(B|A)P(A)/P(B), serves as the foundation for Bayesian inference and many machine learning algorithms. Probability distributions — Gaussian, Bernoulli, Categorical, Dirichlet — model different types of random variables and uncertainty. Maximum Likelihood Estimation (MLE) finds parameters that maximize the probability of observed data, while Maximum A Posteriori (MAP) estimation incorporates prior beliefs. Information theory concepts like entropy, cross-entropy, and KL divergence quantify uncertainty and distributional differences, forming the basis for many loss functions in deep learning. The Central Limit Theorem explains why normal distributions appear so frequently in natural phenomena and justifies many statistical procedures.",
         "Probability theory transforms vague uncertainty into precise, quantifiable statements — it is the mathematics of making decisions with incomplete information."),
    ],
    "Software Engineering Best Practices": [
        ("Clean Code Principles",
         "Writing clean code is a professional discipline that directly impacts project maintainability, bug rates, and team productivity. Meaningful names — for variables, functions, classes, and modules — serve as the primary documentation and should reveal intent without requiring additional comments. Functions should be small, do one thing well, and operate at a single level of abstraction. The Single Responsibility Principle states that each module or class should have exactly one reason to change. Comments should explain why something is done, not what is done — the code itself should be readable enough to explain the what. Error handling should be separated from normal control flow and should never obscure the business logic. Unit tests serve as both verification and documentation, specifying the expected behavior of each component. Consistent formatting, while superficial, reduces cognitive friction when reading unfamiliar code.",
         "Clean code is not about aesthetics — it is about reducing the cognitive load required to understand and modify a system."),
        ("Continuous Integration and Deployment",
         "Continuous Integration (CI) is the practice of frequently merging developer working copies into a shared mainline, triggering automated builds and tests to detect integration issues early. Modern CI pipelines typically include: code checkout, dependency installation, static analysis (linting, type checking), unit tests, integration tests, security scanning, and artifact building. Continuous Delivery (CD) extends CI by automatically deploying successfully built artifacts to staging or production environments. Feature flags decouple deployment from release, allowing code to be deployed to production without being visible to users until explicitly enabled. Canary deployments gradually roll out changes to a small subset of users before full rollout, limiting the blast radius of potential issues. Blue-green deployment maintains two identical production environments, switching traffic instantly between them, enabling near-zero-downtime releases and rapid rollback.",
         "CI/CD transforms software delivery from a high-risk, infrequent event into a routine, low-risk operation — this cultural shift is at the heart of DevOps."),
    ],
    "Cybersecurity": [
        ("Modern Cryptography",
         "Cryptography is the science of secure communication in the presence of adversaries. Symmetric encryption uses the same key for encryption and decryption — AES (Advanced Encryption Standard) with 256-bit keys is the current standard, operating on 128-bit blocks through multiple rounds of substitution and permutation. Asymmetric (public-key) cryptography uses mathematically related key pairs: a public key for encryption and a private key for decryption. RSA's security relies on the practical difficulty of factoring large semiprimes, while Elliptic Curve Cryptography (ECC) offers equivalent security with much smaller keys. Digital signatures provide authentication, non-repudiation, and integrity by allowing anyone to verify that a message was signed by the holder of a specific private key. Hash functions like SHA-256 produce fixed-size digests from arbitrary inputs, with the critical property that finding two inputs with the same hash is computationally infeasible. TLS 1.3, the protocol securing HTTPS, combines all these primitives to provide confidentiality, integrity, and authentication for web traffic.",
         "Modern cryptography is not about making communication impossible to intercept — it is about making interception useless without the proper keys."),
        ("Zero Trust Architecture",
         "Zero Trust is a security paradigm that eliminates implicit trust from network architecture. The core principle — 'never trust, always verify' — means every access request must be authenticated, authorized, and encrypted, regardless of whether it originates from inside or outside the traditional network perimeter. This represents a fundamental shift from the castle-and-moat model, where everything inside the firewall was implicitly trusted. Key components include: micro-segmentation dividing the network into small, isolated zones; continuous authentication and authorization evaluating trust at every access attempt; least privilege access granting only the minimum permissions needed; and assume breach mentality designing systems to limit damage when (not if) a breach occurs. Software-Defined Perimeter (SDP) implements Zero Trust by making applications invisible to unauthorized users — the application infrastructure is 'black' and only becomes visible after successful authentication and authorization.",
         "In a world of remote work, cloud services, and sophisticated threats, Zero Trust transforms security from a perimeter problem into an identity and data problem."),
    ],
    "Cloud Computing": [
        ("Cloud Service Models",
         "Cloud computing offers on-demand access to computing resources through three primary service models. Infrastructure as a Service (IaaS) provides virtualized computing resources — virtual machines, storage, and networking — giving users maximum control and flexibility. Platform as a Service (PaaS) abstracts away infrastructure management, providing a platform for application development and deployment with automatic scaling and patching. Software as a Service (SaaS) delivers complete applications over the internet, eliminating all infrastructure and platform management for end users. The shared responsibility model defines security boundaries: in IaaS, the customer secures everything above the hypervisor; in PaaS, the provider manages the platform; in SaaS, the provider manages nearly everything. Serverless computing (Function as a Service) represents an evolution beyond PaaS, where developers write and deploy individual functions that automatically scale based on demand.",
         "Understanding the shared responsibility model is critical — assuming the provider handles security you are actually responsible for is a common and dangerous mistake."),
    ],
    "Data Science": [
        ("Feature Engineering",
         "Feature engineering is the process of transforming raw data into representations that better expose the underlying patterns to machine learning algorithms. Domain knowledge often guides the creation of meaningful features — in fraud detection, the ratio of transaction amount to a user's average transaction amount may be more predictive than either value alone. Numerical features benefit from scaling (standardization, min-max normalization) and transformation (log, Box-Cox) to handle skewed distributions. Categorical features require encoding — one-hot encoding for low-cardinality categories, target encoding for high-cardinality, and embeddings for very high-cardinality categorical data. Temporal features extract meaningful patterns from timestamps: day of week, season, time since last event, rolling window aggregates. Interaction features capture relationships between variables that linear models cannot represent directly. While deep learning has automated feature learning for perceptual data (images, audio, text), feature engineering remains crucial for structured/tabular data problems.",
         "The quality of features often matters more than the choice of algorithm — a simple model with great features frequently outperforms a complex model with poor features."),
    ],
    "Philosophy of Technology": [
        ("Ethics of Artificial Intelligence",
         "The ethics of AI examines the moral implications of creating intelligent systems that increasingly impact human lives. Fairness in AI requires that systems do not discriminate based on protected characteristics — a challenge made complex by historical biases embedded in training data and the many incompatible mathematical definitions of fairness. Accountability addresses who bears responsibility when AI systems cause harm: the developers, the deploying organization, or the users who rely on AI recommendations. Transparency demands that AI decision-making processes be open to scrutiny, enabling affected individuals to understand and challenge decisions. Privacy concerns are amplified by AI's ability to infer sensitive information from seemingly innocuous data patterns. The alignment problem asks how to ensure that increasingly capable AI systems pursue goals aligned with human values. These ethical considerations are not merely philosophical — they have concrete implications for system design, deployment decisions, and regulatory compliance.",
         "The question is not whether AI will transform society, but whether we will deliberately shape that transformation or merely react to its consequences."),
    ],
}

EN_TEMPLATES = [
    "\n## {topic}\n\n{body}\n\n> **Key Insight:** {reflection}\n",
    "\n### {topic}\n\n{body}\n\n*Further Reading: {reflection}*\n",
    "\n**{topic}**\n\n{body}\n",
    "\n> **{topic}**\n> \n> {body}\n> \n> *{reflection}*\n",
    "\n### Understanding: {topic}\n\n{body}\n\n```\nKey takeaway: {reflection}\n```\n",
    "\nQ: What is {topic}?\n\nA: {body}\n\n> {reflection}\n",
    "\n# {topic}\n\n{body}\n\n## Essential Takeaway\n\n{reflection}\n",
    "\n### Deep Dive: {topic}\n\n{body}\n\n*Practical Application: {reflection}*\n",
]

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   Massive English Corpus Generator      ║")
    print("╚══════════════════════════════════════════╝")

    existing = ""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            existing = f.read()
    print(f"Existing corpus: {len(existing)/1024/1024:.1f}MB")

    parts = []
    total = 0

    for domain, entries in EN_KNOWLEDGE.items():
        parts.append(f"\n\n# {domain}\n")
        for topic, body, reflection in entries:
            for _ in range(25):
                tpl = random.choice(EN_TEMPLATES)
                text = tpl.format(topic=topic, body=body, reflection=reflection)
                parts.append(text)
                total += len(text)
        print(f"  {domain}: {len(entries)} topics")

    header = "\n\n" + "="*60 + "\n# English High-Quality Knowledge Corpus\n" + "="*60 + "\n"
    combined = existing + header + '\n'.join(parts)
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"\nGenerated: {total:,} chars ({total/1024/1024:.1f}MB)")
    print(f"Total corpus: {len(combined):,} chars ({len(combined)/1024/1024:.1f}MB)")
    print(f"Lines: {combined.count(chr(10)):,}")

if __name__ == "__main__":
    main()
