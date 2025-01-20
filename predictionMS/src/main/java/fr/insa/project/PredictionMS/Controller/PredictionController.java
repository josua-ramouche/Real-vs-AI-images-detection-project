package fr.insa.project.PredictionMS.Controller;

import fr.insa.project.PredictionMS.Model.ImagePredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
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

    
    @PostMapping(value = "/image/gradcam", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public MultipartFile predictImage(@RequestParam("image") MultipartFile file) {
        return predictionService.getGradCam(file);
    }
}
