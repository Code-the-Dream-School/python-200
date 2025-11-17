# Setting Up Azure Cloud Shell with Storage
By the end of this activity, you will have a *persistent* Azure Cloud Shell workspace that saves files and SSH keys between sessions. This gives you a safe, browser-based development environment for all cloud work in this course.

By default, the Cloud Shell runs in a *temporary container*. When you close your browser, everything disappears: files, API keys, scripts. To fix that, we'll connect it to a small piece of *persistent storage* called a *file share*. You'll also create an *SSH key pair*, creating a secure way to access online resources in future lessons, without the need for passwords. 

Even this simple task will help us explore the Azure Portal, Cloud Shell, and explore some useful concepts that we will see in more detail in future weeks. 

## 1. Create a Resource Group 
All Azure resources — such as storage accounts and virtual machines — must belong to a *resource group*. A resource group is a container that keeps related resources together (e.g., all the compute and memory resources that go with one project). 

For more on [resource groups](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/overview#resource-groups)

Here, we'll create a resource group to contain resources for storage. 
Note: Resource groups will already be created for you.

1. Sign in at https://portal.azure.com.
2. In the left menu choose **Resource groups  + Create**.
3. Fill in:
   - **Subscription:** your active CTD subscription  
   - **Resource group name:** `cloudshell-rg`  
   - **Region:** your nearest region (for instance, East US)
4. Click **Review + create  Create**.

## 2. Set up persistent storage 
In this section, we're setting up the persistent storage that Cloud Shell needs to save files between sessions. We won't explore how cloud memory and storage work in detail until Week 3, so don't worry if some of these details feel abstract right now. 

### Create a Storage Account 
The storage account is like your own private workspace in your Azure account where different types of data can be stored. It's where you set the configuration parameters for storage. 

1. In the portal, go to **Storage accounts  + Create**.
2. Fill in:
   - **Resource group:** `cloudshell-rg`
   - **Storage account name:** `cloudstore<firstname>` (e.g., `cloudstoreeric`)
   - **Region:** pick the same region as you did for your resource group
   - **Storage type:** *Azure files*
   - **Performance:** *Standard*
   - **Redundancy:** *Locally redundant storage (LRS)*
3. Click **Review + create** -> **Create**. After you click **Create**, you'll see a short progress message at the top of the screen as it creates the Storage Account. 


### Create the File Share
Now that you have a storage account, we will create a *file share*, which is a specific folder inside the storage account. This is the actual directory that you will be able to access. 

We will name the drive `clouddrive`, which is special: Cloud Shell automatically looks for a directory with that name when setting up your persistent storage and mounts it each time you start Cloud Shell.

1. After creating the storage account in the previous step, you can open it by going to Home → Storage accounts → [your storage account name] in the Azure Portal.
2. In the left panel in the Portal, select **Data storage ->  File shares -> + File share** to create a file share.
3. Name it **`clouddrive`** (as mentioned, Cloud Shell expects that).
4. Click **Create**.
5. Open **clouddrive** -> click **edit quota** and se to **5 GB.**
### Mount the File Share in Cloud Shell
We are now ready to move past emphemeral Cloud Shell!

1. In the portal top bar, click the **Cloud Shell ( >_ )** icon.
2. Choose **Bash** when prompted.
3. You'll see a setup panel titled **Mount storage account**.
4. Fill in:
   - **Subscription:** your CTD subscription  
   - **Resource group:** `cloudshell-rg`  (you created this above)
   - **Storage account:** `cloudstore<firstname>`  
   - **File share:** `clouddrive`
5. Click **Apply**.

Azure will display:  
> "Your Cloud Shell is now backed by storage account 'cloudstore<firstname>'."

This means persistence is active! Congrats! You can verify persistence inside the cloud shell with: 

```bash
ls ~/clouddrive
echo "hello cloud shell" > ~/clouddrive/test.txt
ls ~/clouddrive
```

Then close Cloud Shell and reopen it (click the `>_` icon again).  
Run `ls ~/clouddrive` once more  you should still see `test.txt`.  

In general, each time you start the Cloud Shell, the `clouddrive/` directory should be in your home directory. 

## 4. SSH Keys
We have one more important step, which is one of the main points for doing all this work: creating and storing SSH keys. In future weeks, when you connect to a cloud computer (a *virtual machine*), Azure will need to verify who you are. Instead of a password, you'll use an **SSH key pair**. 

With a persistent storage mechanism in place in our shell, we will be able to store SSH keys across sessions!

### What is SSH?
SSH (Secure Shell) is one of the core technologies that makes the modern internet safe. It provides an encrypted way for computers to talk over insecure networks, protecting everything you send — commands, files, credentials — from being seen or altered. SSH replaced older, insecure login methods like telnet by adding encryption and strong cryptographic identity checks. Today it underlies secure connections to millions of servers, cloud platforms, and developer tools such as GitHub.

SSH uses a key pair -- one public and one private -- to prove identity without ever sending a password. Your private key stays on your computer (never share it). Your public key is shared with the systems you want to access. When you connect, SSH verifies that the two match, creating a secure handshake that confirms who you are. In this course, we’ll use SSH keys to authenticate when setting up virtual machines, but the same mechanism also powers secure Git operations and many other online services.

More on [SSH](https://www.cloudflare.com/learning/access-management/what-is-ssh/)

###  Generate SSH Keys
Let's create our SSH keys using the Cloud Shell. As discussed above, you'll actually make two files that work together: a private key and a public key. The private key (`id_rsa`) will stay on your Cloud Shell and should never be shared. The public key (`id_rsa.pub`) is safe to share; Azure will copy it into any virtual machines or services that need to recognize you.

We will store these keys in the default SSH folder at `~/.ssh/`, which is where Azure CLI automatically looks when you try to create new resources using commands like `az create vm`. 

In your cloud shell, first create the directory for the keys and then generate the keys using the `ssh-keygen` command (press Enter to accept defaults):

```bash
mkdir -p ~/.ssh  # create .ssh directory in home
ssh-keygen -t rsa -b 4096 
```

You will be asked where to save, and if you want to create a passphrase, just press Enter to accept the defaults. 

When you run this command, it automatically saves your new keys in the default SSH folder (~/.ssh/). That is, if that folder already exists from the previous step, the keys will be created there automatically, with the private key named `id_rsa` and the public key named `id_rsa.pub`.

Also, once your key is created, you'll see a block of random characters called randomart. This is just a visual fingerprint for your key -- it's normal and nothing you need to worry about. You can think of it as a 1990s-era NFT associated with your SSH key. :smile: 

Go ahead and verify that you created the keys:

```bash
ls ~/.ssh
```

Because your Cloud Shell now has persistence, these keys will remain available in every session.


**Question** Will this disappear because we haven't placed `.ssh` in `clouddrive/`?
**Answer**: While indeed, we only set up your `clouddrive/` folder to persist between sessions, Azure makes an exception for SSH keys. Because the `.ssh` folder is so important, the Cloud Shell automatically detects it and links it to your persistent storage. This ensures your keys are available every time you return to the Cloud Shell!

### Back Up Your Keys Locally
Even though Cloud Shell keeps your keys safe between sessions, it's still a good idea to download a copy to your local computer. If the cloud storage link ever breaks or you create other resources that use your key, having a backup will save you time later.

1. In Cloud Shell, click **manage files** select **Download**.  
2. Enter path `/home/<yourname>/.ssh/`.
3. Download both:
   - `id_rsa`   keep private and secure  
   - `id_rsa.pub`   safe to share or view  

Store them on your computer in a folder such as  
`C:\Users\<you>\.ssh\` (on Windows) or `~/.ssh/` (on macOS/Linux).

## 5. Summing up
Congratulations! You've just set up your persistent Azure Cloud Shell workspace and created your first SSH key pair. That means you now have a secure, browser-based command line that remembers your files and credentials between sessions. This setup will make all your future cloud work easier — especially when you start creating virtual machines and deploying applications in the next lessons.

add cleanup step