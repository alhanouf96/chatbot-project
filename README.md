Project Overview
Objectives
In this project, we will first develop a chatbot locally. Then, we will deploy the application on Azure using three different architectures:

1. Server-based Architecture

2. Serverless Architecture

3. Containerized Architecture

Milestone and Stages
Local Application Development(Stage 1-4)
Description:

At this stage, we will develop an RAG chatbot in the local environment. We will start with the basic Streamlit chatbot and split the front and back ends. Then, we will add features like chat history storage and RAG. At the end of these stages, we will have a RAG chatbot that can ask questions about PDF documents.

stage1-4
Stage breakdown:

After learning the Streamlit, we can start to build our basic chatbot. (Stage 1)
After learning the FastAPI, we can do the front and back end splitting (Stage 2)
Once we finish the front-end and back-end splitting, we can connect to the database and add the feature to store and load the chat history. (Stage 3)
Then, at the end, we will add the RAG feature. (Stage 4)
Finally, we need to push the codes to the GitHub repo.
Milestone 1: Server-based Deployment (Stage 5-6)
Description:

At the end of Milestone 1, we should be able to use Terraform to provision the following resources: a VMSS to host our application, a VM to host our vector database, an Azure Database for PostgreSQL to store data, an Azure Storage Account to store files, an Azure Application Gateway to redirect traffic, and an Azure Key Vault to store secrets and environment variables



Stage breakdown:

After learning the Azure VM, we need to deploy the application on it (clone them from the Github repo) and set up the database. There are no changes on the code side. (Stage 5)
After learning Azure Blob Storage, we need to update the codes correspondingly to save the chat history and PDF files to the Blob Storage. (Stage 6)
After learning Azure PostgreSQL, they can move the database from VM to Azure PostgreSQL. There are no changes on the code side. All we need to do is change the .env file. (Stage 6)
After learning Azure Key Vault, we can move most of our credentials from the .env file to the Key Vault. (Stage 6.5)
Infrastructure Setup:

Stage 5:
We have to manually launch a VM on Azure and run our application on it. 
Once we have learnt Terraform, we will build the stage 5 infrastructure with it.
stage5

Stage 6:
We have to manually add an Azure Storage account, and an Azure Database for PostgreSQL on Azure. 
Once the application is running on the VM, we will configure a GitHub Action workflow to deploy our application to the VM continuously. This is due by week 4
Once we have learnt Terraform, we will launch stage 6 infrastructure with it. This is due by week 6
stage6

Stage 6.1:
We have to manually add a VMSS to host our application and an Application Gateway to redirect traffic. 
Once the application is running on the VMSS, we will configure a GitHub Action workflow to deploy our application to the VM continuously. This is due by week 5
Once we have learnt Terrafrom, we will launch stage 6.1 infrastructure with it. This is due by week 8
stage6.1

Stage 6.5:
We have to manually add an Azure Key Vault to store the secrets and environment variables. It is due by week 7.
Once the application is running on the VMSS, we will configure a GitHub Action workflow to deploy our application to the VM continuously. This is due by week 7
Once we have learnt Terrafrom, we will launch stage 6.5 infrastructure with it. This is due by week 9
stage6.5

Stage 6.6:
We launch 2 identical sets of infrastructure as shown in the diagram with Terraform. One can be used as our dev environment. The other one can be used as our production environment. 

stage6.6

Milestone 2: Serverless Deployment (Stage 7-9)
Description:

At the end of Milestone 2, all the back-end codes will be deployed on the Azure Function, and the chat history will be stored on the Azure CosmosDB.



Stage breakdown:

After learning the Azure Function, we can migrate our codes from FastAPI to the Azure Function. In this way, we no longer need to run the backend on the VM. The backend will be deployed using a serverless approach. (Stage 7)
After learning the CosmosDB, we no longer need to save the chat history in the Blob Storage. Instead, we can save it to the CosmosDB. (Stage 8)
Since Binding is a special feature of Azure Function, we can also apply it in our project, for example, we can use it to store the PDF file in the blob storage. (Stage 9)
Infrastructure Setup:

Stage 7:
We launch the infrastructure as shown in the diagram with Terraform. In this architecture, the backend is hosted on Azure Function Apps. The VM is used to host the frontend and the streamlit. 
A new CICD pipeline will be built so that when there’s an update to the backend, the change should be deployed to the Function Apps automatically.
stage7

Stage 8:
We launch the infrastructure as shown in the diagram with Terraform. A CosmosDB is added compared with the previous stage. 
stage8

Milestone 3: Containerized Deployment (Stage 10)
Description:

By the end of this Milestone, students will be able to dockerize the functions and deploy the docker image on the Azure Container App.



Stage breakdown:

After learning the Docker and Azure Container App, we can dockerize our FastAPI codes and deploy them on the Azure Container App.
Infrastructure Setup:

Stage 10:
We launch the infrastructure as shown in the diagram with Terraform. In this architecture, the backend is hosted on Azure Container Apps. The images of the backend are stored in the Azure Container Registry. The VM is used to host the frontend and the streamlit. 
A new CICD pipeline will be built so that when there’s an update to the backend, the change should be deployed to the Azure Container Apps automatically.
stage10
