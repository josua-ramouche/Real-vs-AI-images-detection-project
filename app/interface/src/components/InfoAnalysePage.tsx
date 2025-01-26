
import { Alert, Button } from "@mui/material";
import axios from "axios";
import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import DisplayCard from "./DisplayCard";
import { MagnifyingGlass } from "react-loader-spinner";


const InfoAnalysePage: React.FC = () => {
    const location = useLocation();
    const API_BASE_URL = 'http://192.168.37.156:50000/predict/image/gradcam';
    const [loading, setLoading] = useState(false);
    const { file, fileURL , result, accuracy} = location.state || {};
    const [grad, setgrad] = useState<File | null>(null);
    const [gradURL, setgradURL] = useState<string | null>(null);
    const navigate = useNavigate();


    useEffect(() => {
            const handleDisplayResults = async () => {
                if (!file) return;
    
                setLoading(true);
                const formData = new FormData();
                formData.append('image', file);
    
                try {
                    const response = await axios.post(API_BASE_URL, formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data',
                        },
                        responseType: 'blob',
                    });
    
                    const data = response.data;

                    if (data instanceof Blob && data.type.startsWith('image/')) {
                        const imageUrl = URL.createObjectURL(data);
                        setgradURL(imageUrl);
                    } else {
                        console.error("Unexpected response structure:", data);
                    }
                } catch (error) {
                    console.error('Error uploading file:', error);
                } finally {
                    setLoading(false);
                }
            };
    
        handleDisplayResults();
    }, [file]);

    const handleNavigateToResults = () => {
        if (file && fileURL && result && accuracy) {
            const r = result;
            const a = accuracy;
            navigate('/results', {
            state: { file, fileURL, r, a },
            });
        }
        };

    return(
        <div className="page">
            <div className="instructions">
                <p> Grad-CAM of the result </p>
            </div>
            {loading ? (
                <MagnifyingGlass color="#65b2a0" />
            ) : (
                <div className="page">
                    {gradURL && <img src={gradURL} alt="Grad-CAM" style={{ maxWidth: "100%", maxHeight: "300px" }} />}
                    <Alert severity="info" sx={{marginBottom: "20px", marginTop: "20px", width: "750px", flexGrow: 1}}>
                        A Grad-CAM is a graphical representation used in Deep Learning to visualize and understand the decision made by the model.
                        It generates an heatmap that highlights the crucial regions of an image.
                        The red areas are the pixels that help the most the model in making the decision and the blue ones are the less important.
                    </Alert>
                    <DisplayCard result={result} accuracy={accuracy}/>
                </div>
            )}
            <Button 
            variant="contained"
            onClick={handleNavigateToResults}
            sx={{
              marginTop: "20px",
              backgroundColor: "#65b2a0",
              color: "#fff",
              "&:hover": { backgroundColor: "#386258" },
            }}>
                Go back
            </Button>
        </div>
    );

}

export default InfoAnalysePage;