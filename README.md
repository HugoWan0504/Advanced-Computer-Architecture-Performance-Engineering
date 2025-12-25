# Advanced Computer Architecture & Performance Engineering

## Overview

This project investigates **performance-critical aspects of modern computer architectures**, with a focus on how **CPU pipelines, memory hierarchies, and parallelism mechanisms** impact real-world workloads. The work combines **analytical performance modeling** with **hands-on experimentation**, including case studies on **large language model (LLM) inference bottlenecks**.

The repository emphasizes understanding *why* systems behave the way they do—not just how to optimize them.

---

## Core Themes

### Performance Modeling

* Applied the **performance equation** to reason about execution time in terms of instruction count, CPI, and clock rate
* Quantified the impact of **instruction-level parallelism (ILP)**, pipeline hazards, and branch behavior on performance

---

### Memory Hierarchy & Cache Behavior

* Analyzed **multi-level cache hierarchies**, including capacity, conflict, and compulsory misses
* Explored how **access patterns, stride, and data layout** affect cache utilization and memory latency
* Investigated cache behavior through controlled experiments and analytical reasoning

---

### Exploiting Modern Processor Features

* Studied architectural features such as **out-of-order execution, SIMD/vectorization, and hardware prefetching**
* Evaluated how compiler and hardware optimizations interact with program structure

---

### Parallelism & Multiprocessing

* Explored **thread-level parallelism (TLP)** and multicore execution models
* Analyzed synchronization overhead, scalability limits, and Amdahl’s Law effects
* Designed experiments to understand parallel speedup and bottlenecks

---

## LLM Performance Case Studies

A dedicated set of notebooks focuses on **LLM inference performance**, applying architectural principles to modern AI workloads:

* Identified **critical performance bottlenecks** in LLM inference pipelines
* Analyzed **KV-cache access patterns** and their interaction with memory hierarchy
* Explored **parallel architectures and data partitioning strategies** to improve throughput and efficiency

---

## Repository Structure

* `A1–A5`: Architecture-focused studies on performance equations, caches, processor features, and parallelism
* `P1–P3`: Applied performance analysis and optimization of **LLM inference workloads**
* Each notebook contains:

  * Conceptual explanations
  * Experimental setups
  * Quantitative analysis and interpretation

---

## Skills Demonstrated

* Computer architecture performance analysis
* Memory hierarchy reasoning and cache optimization
* Parallel performance modeling
* Systems-level understanding of AI workload behavior
* Experimental design and result interpretation

---

## Project Status

* All analyses completed and documented
* Experiments validated through repeated runs
* No known inconsistencies in reported results

---

## Author

**Hugo Wan**

---

## Notes

This project reflects **advanced systems thinking**, bridging **computer architecture fundamentals** with **modern AI performance challenges**, and is representative of skills used in **systems engineering, performance optimization, and hardware–software co-design**.
