# Face-Matching
Given Two images of a particular person of different age, tell whether that person is the same person or not.


## Detailed Explanation of my Code.
The very first thing which I have done is, finding and extracting faces from photos using MTCNN(Multi-task Cascaded Convolutional Network) extract_face, then I got the bounding box coordinates, and using the coordinates I cropped the image, such that at the end I will have only the numpy array of the FACE(Region of Interest).

After extracting the face from the photos, I used get_embedding for transfer learning through FaceNet model for face-verification.

This involves calculating a face embedding for a new given face and comparing the embedding to the embedding for the single example of the face known to the system.
We can use Euclidean distance or Cosine Distance to measure the distance between the face embeddings and the faces are said to be matched(is_match) if the distance is within a specific threshold limit(in my case it's 0.55).
