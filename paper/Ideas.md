---
title: Ideas
tags:
  - paper
  - idea
categories: 
date created: 2024-10-04 18:30:32
date modified: 2024-10-05 16:21:37
---
**How should software systems be designed to support the full machine learning lifecycle, from programming interfaces and data preprocessing to output interpretation, debugging and monitoring?**
- How can we enable users to quickly "program" the modern machine learning stack through emerging interfaces such as manipulating or labeling training data, imposing simple priors or constraints, or defining loss functions?
- How can we enable developers to define and measure ML models, architectures, and systems in higher-level ways?
- How can we support efficient development, monitoring, interpretation, debugging, adaptation, tuning, and overall maintenance of production ML applications - including not just models, but the data, features, labels, and other inputs that define them?

**How should hardware systems be designed for machine learning?**
- How can we develop specialized, heterogeneous hardware for training and deploying machine learning models, fit to their new operation sets and data access patterns?
- How can we take advantage of the stochastic nature of ML workloads to discover new trade-offs with respect to precision, stability, fidelity, and more?
- How should distributed systems be designed to support ML training and serving?

**How should machine learning systems to be designed to satisfy metrics beyond predictive accuracy such as power and memory efficiency, accessibility, cost, latency, privacy, security, fairness, and interpretability?**
- How can machine learning algorithms and systems be designed for device constrains such as power, latency, and memory limits?
- How can ML systems be designed to support full-stack privacy and security guarantees, including, e.g., federated learning and other similar settings?
- How can we increase the accessibility of ML, to empower an increasingly broad range of users who may be neither ML nor systems experts?


- 部署问题：随着机器学习在越来越多样化和任务关键性的方式中得到使用，一系列系统范围内的关切变得越来越普遍。这些包括对抗性影响或其他虚假因素的鲁棒性；更广泛考虑的安全性；隐私和安全性，尤其是随着敏感数据的越来越多使用；可解释性，因为法律和操作上越来越需要；公平性，因为机器学习算法开始对我们的日常生活产生重大影响；以及许多其他类似的关切。
- 成本：最初默认的解决方案是在ImageNet上学习CNN，其计算成本为2300美元，训练时间为13天。仅对大量训练数据进行标注的成本就可能高达数十万到数百万美元。以其他指标（如延迟或功率）衡量，降低成本对于越来越多的设备和生产部署配置也至关重要。
- 可访问性：随着越来越多的人急于将机器学习用于实际生产目的——包括大学里大型新项目培养的一批新一代多语言数据科学家——机器学习系统需要能够被没有博士学位水平的机器学习和系统专业知识的开发者和组织使用。



[9 libraries for parallel & distributed training/inference of deep learning models | by ML Blogger | Medium](https://medium.com/@mlblogging.k/9-libraries-for-parallel-distributed-training-inference-of-deep-learning-models-5faa86199c1f#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjFkYzBmMTcyZThkNmVmMzgyZDZkM2EyMzFmNmMxOTdkZDY4Y2U1ZWYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDg3MzIyNjk2MzU3NjU2NDYyMTIiLCJlbWFpbCI6Imhlc2VueWFuZzAyQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYmYiOjE3MzEzMTQ0MDEsIm5hbWUiOiJoZXNlbiB5YW5nIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0pCaW0yZmdxZlByUWFWeFNfVzgtS0pPVzFvRGlKb2l6YU5XOFEtaDV2bzM3d0NFZz1zOTYtYyIsImdpdmVuX25hbWUiOiJoZXNlbiIsImZhbWlseV9uYW1lIjoieWFuZyIsImlhdCI6MTczMTMxNDcwMSwiZXhwIjoxNzMxMzE4MzAxLCJqdGkiOiJkY2Y1MmI3OTI4YmIyZDJiNTE0ZDg3YzUwNjU4ZjZmZDA1MGYzY2ZlIn0.QgUxrZ8frGSSK5g7ckQJETj-MZYYXJA1MSAnejVeOEyyUSwr4wtc8KYtWXxfQ5yiEkXENrNptVwlSTPUVQTKKuR-cl03TcW2UCZFRNFskuCByRJN3t_qkRc88iXmjB67gWnKI8ChCifneYK65iTU8-zOM2a3POpzrqQZ5YUZ7c7cx5BxQbp-okm59X6emXkZ1b9Z2vxDEa9rATjEVO-8gy0yv5Gzq-QycWKzVOf796ITp6eg3TMwbo6xr3MT7QEzfYQ8r-5rcBEetcDViN5LnQOnXdRF9iFU4-X_ftNVPKrqdYDH0xG0RqNFS8uT5DkG9o6KduOWZaRQUnxZOBhH8A)




