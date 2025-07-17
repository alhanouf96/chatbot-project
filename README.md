
# Azure Chatbot Project Overview

## Objectives

This project aims to build a **Retrieval-Augmented Generation (RAG) Chatbot**, starting from local development and deploying on **Microsoft Azure** using three deployment models:

* **Server-based Deployment**
* **Serverless Deployment**
* **Containerized Deployment**

---

## Development Stages

### 1️⃣ Local Development

* Build a basic chatbot with **Streamlit**.
* Split frontend (Streamlit) and backend (FastAPI).
* Add database integration for chat history.
* Implement RAG functionality to interact with PDFs.

---

### 2️⃣ Server-based Deployment (with Terraform)

* Host application on **Azure VM** and **VMSS**.
* Use **Azure PostgreSQL** for database, **Blob Storage** for files, and **Key Vault** for secrets.
* Set up **Azure Application Gateway** and configure CI/CD with GitHub Actions.

---

### 3️⃣ Serverless Deployment

* Migrate backend to **Azure Functions**.
* Store chat history in **Azure Cosmos DB**.
* Use Azure Bindings for file storage in Blob Storage.

---

### 4️⃣ Containerized Deployment

* Dockerize backend services.
* Deploy containers to **Azure Container Apps** with images stored in **Azure Container Registry**.
* Configure automated deployment pipelines.

---

## Infrastructure Provisioning

All cloud resources will be provisioned using **Terraform**, and CI/CD pipelines will be implemented for continuous deployment to Azure.


