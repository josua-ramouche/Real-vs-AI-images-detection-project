package fr.insa.ImageMS;

import org.bson.types.ObjectId;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.core.io.InputStreamResource;
import org.springframework.data.mongodb.gridfs.GridFsResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

import java.io.IOException;

@RestController
@RequestMapping("/files")
public class ImageController {

    @Autowired
    private DatabaseController fileService;

    // Endpoint to upload image
    @PostMapping("/upload")
    public ResponseEntity<String> uploadFile(@RequestParam("file") MultipartFile file) {
        try {
            ObjectId fileId = fileService.uploadFile(file);
            return ResponseEntity.ok("File uploaded successfully with ID: " + fileId.toString());
        } catch (IOException e) {
            return ResponseEntity.status(500).body("File upload failed");
        }
    }

    // Endpoint to retrieve image by file ID
    @GetMapping("/download/{fileId}")
    public ResponseEntity<InputStreamResource> downloadFile(@PathVariable String fileId) {
        try {
            GridFsResource resource = fileService.downloadFile(fileId);
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=" + resource.getFilename())
                    .contentType(MediaType.IMAGE_JPEG)  // adjust the content type based on your file type
                    .body(new InputStreamResource(resource.getInputStream()));
        } catch (Exception e) {
            return ResponseEntity.status(404).body(null);
        }
    }
    
}
