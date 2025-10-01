# **Week_2 — Hands-On with Microsoft Azure**

## **Lesson Overview**  
**Learning objective:** In this lesson, students will gain their first hands-on experience with Microsoft Azure. They will log into the Azure Portal, explore its main features, launch Cloud Shell, run filesystem and Azure CLI commands, and write a Python script directly in the cloud. By the end of this lesson, students will feel comfortable navigating Azure and executing basic tasks using both the portal and CLI.  

### **Topics:**  
1. Introduction to the Azure Portal  
2. Logging in & Exploring the Portal  
3. Getting Started with Cloud Shell  
4. Exploring the Cloud Shell Filesystem  
5. Running Azure CLI Commands  
6. Working with the Cloud Shell Editor  
7. Executing Python in the Cloud  
8. Connecting with VS Code Desktop (Optional)  
9. Wrap-Up & Next Steps  

---

## **2.1 Introduction to the Azure Portal**

### **Overview**  
The Azure Portal is a **web-based dashboard** for managing Azure services. It enables you to:  
- Create and monitor resources (VMs, storage, databases).  
- Configure networking and security.  
- Manage subscriptions and billing.  

### **Real-World Use Cases:**  
- **Developers:** Deploy and scale applications.  
- **IT Teams:** Manage identity, access, and security.  
- **Businesses:** Ensure reliability, compliance, and scalability.  

🔗 [Azure Portal Overview](https://learn.microsoft.com/en-us/azure/azure-portal/azure-portal-overview)  

---

## **2.2 Logging in & Exploring the Portal**

### **Steps:**  
1. Open [https://portal.azure.com](https://portal.azure.com).  
2. Sign in with your Microsoft account (work, school, or sandbox).  
3. Explore the key UI areas:  
   - **Search bar** — Quickly find any service.  
   - **Create a resource** — Add new Azure resources.  
   - **Dashboard** — Customizable home page.  
   - **Cloud Shell icon** — Access Cloud Shell from the top menu.  

**Activity:** Search for *“Virtual Machine”* and open the service page.  

---

## **2.3 Getting Started with Cloud Shell**

### **Overview**  
Azure Cloud Shell provides a browser-based command-line interface.  

### **Steps:**  
1. Click the **Cloud Shell icon** in the portal header.  
2. Select **Bash** when prompted.  
3. Accept the prompt to create storage (saves your files) or continue without storage.  
4. Select your subscription from the drop-down.  

### **Test the Environment:**  
```bash
az --version
python3 --version
```  

---

## **2.4 Exploring the Cloud Shell Filesystem**

### **Example Commands:**  
```bash
pwd                       # show current directory
ls -la                    # list files
mkdir azure-lab           # create a folder
cd azure-lab
echo "Welcome to Azure Cloud Shell" > hello.txt
cat hello.txt
```  

**Outcome:** Students will be comfortable navigating and managing files within the cloud-based filesystem.  

---

## **2.5 Running Azure CLI Commands**

### **Overview**  
The **Azure CLI** (`az`) is a command-line tool to manage Azure resources programmatically.  

### **Example Commands:**  
```bash
az account show --output table     # show current subscription
az account list --output table     # list all subscriptions
az group list --output table       # list existing resource groups
az vm --help                       # get VM-related help
```  

🔗 [Azure CLI Documentation](https://learn.microsoft.com/en-us/cli/azure/get-started-with-azure-cli?view=azure-cli-latest)  

---

## **2.6 Working with the Cloud Shell Editor**

### **Steps:**  
1. Open the editor in Cloud Shell:  
   ```bash
   code .
   ```  
   *(Confirm the prompt to switch to classic shell if asked.)*  

2. Create a new Python file:  
   ```bash
   touch hello_azure.py
   code .
   ```  

3. Paste the script below into `hello_azure.py`:  
   ```python
   # hello_azure.py
   import os, sys, platform

   print("Hello from Azure Cloud Shell!")
   print("Python version:", sys.version.split()[0])
   print("Platform:", platform.platform())
   print("Current directory:", os.getcwd())
   print("Files:", os.listdir("."))
   ```  

4. Save the file.  

---

## **2.7 Executing Python in the Cloud**

### **Run the Script:**  
```bash
python hello_azure.py
```  

**Outcome:** The script prints system details and file information from within the Azure Cloud Shell environment.  

---

## **2.8 Connecting with VS Code Desktop (Optional)**

### **Steps:**  
1. Install **Visual Studio Code**.  
2. Add the **Azure Tools extension**.  
3. Sign in with your Azure account.  
4. Open Cloud Shell directly from VS Code’s terminal menu.  

---

## **2.9 Wrap-Up & Next Steps**

### **You just accomplished:**  
- Logged into the Azure Portal.  
- Explored the Portal UI.  
- Used Cloud Shell to run filesystem and Azure CLI commands.  
- Created and ran a Python script in the cloud.  

### **Next Steps:**  
- Deploy a Virtual Machine.  
- Compare working with the **Portal GUI vs. Azure CLI**.  
- Keep practicing — Azure skills grow with hands-on experience!  

---

## **Summary**  

In this lesson, you learned:  
1. How to log in and explore the Azure Portal.  
2. How to launch and use Cloud Shell.  
3. How to run filesystem and Azure CLI commands.  
4. How to create and run a Python script in the cloud.  
5. (Optional) How to connect Azure with VS Code.  

---

### **Additional Resources** 

1.  [Azure Getting Started](https://azure.microsoft.com/en-us/get-started)
2. [Azure Portal Overview](https://learn.microsoft.com/en-us/azure/azure-portal/azure-portal-overview)  
2. [Azure CLI Documentation](https://learn.microsoft.com/en-us/cli/azure/get-started-with-azure-cli?view=azure-cli-latest)  
3. [VS Code + Azure Integration](https://code.visualstudio.com/docs/azure/extensions)