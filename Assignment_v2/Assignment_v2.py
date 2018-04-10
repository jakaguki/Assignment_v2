
import cv2, os, shutil
import numpy as np

    
def detect_and_save_faces(name, roi_size):
    
    # define where to look for images and where to save the detected faces
    dir_images = "data/{}".format(name)
    dir_faces = "data/{}/faces".format(name)
    if not os.path.isdir(dir_faces): os.makedirs(dir_faces)  
    
    # put all images in a list
    names_images = [name for name in os.listdir(dir_images) if not name.startswith(".") and name.endswith(".jpg")] # can vary a little bit depending on your operating system
    
    # detect for each image the face and store this in the face directory with the same file name as the original image
    ## TODO ##
    face_cascade = cv2.CascadeClassifier(os.path.join("data/",'haarcascade_frontalface_alt.xml'))
    for i in names_images:
        img = cv2.imread(os.path.join(dir_images,i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray)[0]
        #Rect(x,y,w,h)
        face_img = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]

        cv2.imwrite(os.path.join(dir_faces,i),cv2.resize(face_img,roi_size))
           
    
def do_pca_and_build_model(name, roi_size, numbers):
    
    # define where to look for the detected faces
    dir_faces = "data/{}/faces".format(name)
    
    # put all faces in a list
    names_faces = ["{}.jpg".format(n) for n in numbers]
    
    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    ## TODO ##
    N = len(names_faces)
    P = roi_size[0]*roi_size[1]
    X = np.zeros((N,P))
    iterate = 0
    for i in numbers:
        img = cv2.imread(os.path.join(dir_faces,"{}.jpg".format(i)),0)
        k = np.reshape(img,(1,P))
        #X[i-1,:] = k
        X[iterate,:] = k
        iterate += 1
    
        
    #k = np.zeros((roi_size[0],roi_size[1]))
    #for i in range(np.shape(X)[0]):
    #    k = k+np.reshape(X[i,:],(roi_size[0],roi_size[1]))

    #k = scale_to(k,255)
    #cv2.imwrite('k.jpg',k)
    #print(np.min(np.min(k,axis = 0)))
    #print(np.max(np.max(k,axis = 0)))
    # calculate the eigenvectors of X
    #P-re lett átírva a number of components
    mean, eigenvalues, eigenvectors = pca(X, number_of_components=12)
    
    return [mean, eigenvalues, eigenvectors]
    

def test_images(name, roi_size, numbers, models):

    # define where to look for the detected faces
    dir_faces = "data/{}/faces".format(name)
    
    # put all faces in a list
    names_faces = ["{}.jpg".format(n) for n in numbers]
    
    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    ## TODO ##
    N = len(names_faces)
    P = roi_size[0]*roi_size[1]
    X = np.zeros((N,P))

    for i in range(np.shape(X)[0]):
        k = names_faces[i]
        img = cv2.imread(os.path.join(dir_faces,names_faces[i]),0)
        k = np.reshape(img,(1,P))
        X[i-1,:] = k
        z= np.shape(X)
        
    # reconstruct the images in X with each of the models provided and also calculate the MSE
    # store the results as [[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    results = []
    for model in models:
        projections, reconstructions = project_and_reconstruct(X, model)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        results.append([reconstructions, mse])

    return results
    

def pca(X, number_of_components):
    
    R = np.transpose(X)
    mean = np.mean(R, axis = 1)
    covariance = np.cov(R)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    eigenvectors = np.real(eigenvectors)
    eigenvalues = np.real(eigenvalues)

    eigenvalues = eigenvalues[:number_of_components]
    eigenvectors = eigenvectors[:,:number_of_components]

    print(np.shape(eigenvectors))
    ## TODO ##
    
    return [mean, eigenvalues, eigenvectors]


def project_and_reconstruct(X, model):


    projections = np.dot(np.transpose(model[2]),np.transpose(X-model[0]))
    k = model[2]
    z = projections
    reconstructions = np.dot(model[2],projections)
    reconstructions = np.transpose(reconstructions)+model[0]
    ## TODO ##

    
    return [projections, reconstructions]

def visualize_model(model, name):
    dir_model = "data/{}/model".format(name)
    if not os.path.isdir(dir_model): os.makedirs(dir_model)
    name  = name + "_model"
    img = np.zeros((roi_size[0],roi_size[1]))
    for i in range(np.shape(model[2])[1]):
        k = model[2][:,i]
        img = img + np.reshape(model[2][:,i]*model[1][i],(roi_size[0],roi_size[1]))

    img = scale_to(img,250)
    cv2.imwrite(os.path.join(dir_model,"{}.jpg".format(name)),img)

def scale_to(x,b):
    #scales the thing to (0,b)
    mi = np.min(np.min(x,axis = 0)) 
    x = x+np.ones_like(x)*(-mi)
    ma = np.max(np.max(x,axis = 0))
    x = np.round(x/ma*b)
    return x

def visualize_result(result, name):
    
    print(len(result[0][1]))
    for i in range(len(result)):
        dir_results = "data/{}/results/{}".format(name,i)
        if not os.path.isdir(dir_results): os.makedirs(dir_results)
        file = open("data/{}/results/{}/file.txt".format(name,i),'w') 
        for j in range(len(result[i][0])):
            img = np.reshape(result[i][0][j,:],(roi_size[0],roi_size[1]))
            #cv2.imwrite(os.path.join(dir_results,"{}.jpg".format(result[i][1][j])),img)
            cv2.imwrite(os.path.join(dir_results,"{}.jpg".format(j)),img)
        sum = 0
        for j in range(len(result[i][0])):
            file.write(str(result[i][1][j]))
            file.write('\n')
            sum = sum + result[i][1][j]
        file.write('\n')
        file.write(str(sum/len(result[i][0])))
        file.close()

def visualize_rois(name,roi_size):
    # define where to look for images and where to save the detected faces
    dir_images = "data/{}".format(name)
    dir_rois = "data/{}/rois".format(name)
    if not os.path.isdir(dir_rois): os.makedirs(dir_rois)  
    
    # put all images in a list
    names_images = [name for name in os.listdir(dir_images) if not name.startswith(".") and name.endswith(".jpg")] # can vary a little bit depending on your operating system
    
    # detect for each image the face and store this in the face directory with the same file name as the original image
    ## TODO ##
    face_cascade = cv2.CascadeClassifier(os.path.join("data/",'haarcascade_frontalface_alt.xml'))
    for i in names_images:
        img = cv2.imread(os.path.join(dir_images,i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray)[0]
        #Rect(x,y,w,h)
        img = cv2.rectangle(img,(face[0],face[1]), (face[0]+face[2],face[1]+face[3]),(255,0,255),2)
        cv2.imwrite(os.path.join(dir_rois,i),img)
        
def clear_old_results(name):
    dirs = []
    dirs.append( "data/{}/faces".format(name))
    dirs.append("data/{}/rois".format(name))
    dirs.append("data/{}/results".format(name))
    dirs.append("data/{}/model".format(name))
    for i in dirs:
        shutil.rmtree(i)

if __name__ == '__main__':
    
    #clear_old_results("arnold")
    #clear_old_results("barack")

    roi_size = (30, 30) # reasonably quick computation time
    
    # Detect all faces in all the images in the folder of a person (in this case "arnold" and "barack") and save them in a subfolder "faces" accordingly
    detect_and_save_faces("arnold", roi_size=roi_size)
    detect_and_save_faces("barack", roi_size=roi_size)
    
    # visualize detected ROIs overlayed on the original images and copy paste these figures in a document 
    ## TODO ## # please comment this line when submitting
    visualize_rois("arnold", roi_size)
    visualize_rois("barack", roi_size)

    
    # Perform PCA on the previously saved ROIs and build a model=[mean, eigenvalues, eigenvectors] for the corresponding person's face making use of a training set
    model_arnold = do_pca_and_build_model("arnold", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])
    #model_barack = do_pca_and_build_model("barack", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])
    model_barack = do_pca_and_build_model("barack", roi_size=roi_size, numbers=[1, 2, 4, 5, 6])
    
    # visualize these "models" in some way (of your choice) and copy paste these figures in a document
    ## TODO ## # please comment this line when submitting
    visualize_model(model_arnold, "arnold")
    visualize_model(model_barack, "barack")
    
    # Test and reconstruct "unseen" images and check which model best describes it (wrt MSE)
    # results=[[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    # The correct model-person combination should give best reconstructed images and therefor the lowest MSEs
    results_arnold = test_images("arnold", roi_size=roi_size, numbers=[7, 8], models=[model_arnold, model_barack])
    #results_barack = test_images("barack", roi_size=roi_size, numbers=[7, 8, 9, 10], models=[model_arnold, model_barack])
    results_barack = test_images("barack", roi_size=roi_size, numbers=[7, 8, 10], models=[model_arnold, model_barack])


    # visualize the reconstructed images and copy paste these figures in a document
    ## TODO ## # please comment this line when submitting
    visualize_result(results_arnold,"arnold")
    visualize_result(results_barack,"barack")

