package fr.insa.ImageMS;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.gridfs.GridFsTemplate;
import org.springframework.data.mongodb.gridfs.GridFsResource;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.stereotype.Service;
import java.io.IOException;
import org.bson.types.ObjectId;

@Service
public class DatabaseController {
    
    @Autowired
    private GridFsTemplate gridFsTemplate;

    // Upload the image to MongoDB
    public ObjectId uploadFile(MultipartFile file) throws IOException {
        return gridFsTemplate.store(file.getInputStream(), file.getOriginalFilename());
    }

    // Download the image from MongoDB
    public GridFsResource downloadFile(String fileId) {
        return gridFsTemplate.getResource(fileId);
    }
}
