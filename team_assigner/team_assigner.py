from sklearn.cluster import KMeans

class TeamAssigner:

    def __init__(self):
        self.team_colors = {}

    def get_clustering_model(self, image):
        #reshape the image to a 2D array of pixels
        image_2D = image.reshape(-1, 3)

        # Use KMeans clustering to find the dominant colors in the image
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2D)

        return kmeans

    def get_player_color(self, frmae, bbox):
        # Extract the player's color from the frame
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        top_half_image = image[:int(image.shape[0]/2), :]

        # Get the dominant color of the player's jersey
        kmeans = self.get_clustering_model(top_half_image)

        #get the cluster labels for each pixel in the image
        labels = kmeans.labels_

        #reshpa the labels to the shape of the image
        clustred_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        #get the player cluster
        corner_clusters = [[clustred_image[0, 0], clustred_image[0, -1]], [clustred_image[-1, 0], clustred_image[-1, -1]]]
        player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Use KMeans clustering to find the team colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(player_colors)

        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]