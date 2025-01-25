package fr.insa.project.PredictionMS.Controller;

import fr.insa.project.PredictionMS.Model.ImagePredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/predict")
public class PredictionController{

    @Autowired
    private ImagePredictionService predictionService;

    @PostMapping(value = "/image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public String predictImage(@RequestParam("image") MultipartFile file) {
        return predictionService.getPrediction(file);
    }


    @PostMapping("/image/gradcam")
    public ResponseEntity<byte[]> generateGradCam(@RequestParam("image") MultipartFile file) {
        try {
            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_PNG)
                    .body(predictionService.getGradCam(file));
        } catch (Exception e) {
            return null;
        }
    }

}
