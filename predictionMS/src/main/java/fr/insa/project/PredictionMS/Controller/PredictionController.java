package fr.insa.project.PredictionMS.Controller;

import fr.insa.project.PredictionMS.Model.ImagePredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/predict")
public class PredictionController{

    @Autowired
    private ImagePredictionService predictionService;

    @PostMapping("/image")
    public String predictImage(@RequestParam("image") MultipartFile file) {
        return predictionService.getPrediction(file);
    }
}
