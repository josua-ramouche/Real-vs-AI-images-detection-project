import { Alert, Button } from "@mui/material";
import { MagnifyingGlass } from "react-loader-spinner";
import DisplayCard from "./DisplayCard";
import InfoIcon from '@mui/icons-material/Info';
import { useEffect, useState } from "react";
import axios from "axios";
import { useLocation, useNavigate } from "react-router-dom";


const TypeAlertMessageEnum = {
    SuccessMessage: 'Given the accuracy, this result is very reliable!',
    InfoMessage: 'Given the accuracy, this result is fairly reliable',
    WarningMessage: 'Given the accuracy, this result may be uncertain',
    ErrorMessage: 'Given the accuracy, this result is uncertain',
} as const;

const ResultsPage: React.FC = () => {
    const location = useLocation();
    const { file, fileURL, r, a } = location.state || {};
    const [result, setResult] = useState<string | null>(r);
    const [accuracy, setAccuracy] = useState<number | null>(a);
    const [loading, setLoading] = useState(false);
    const [alertMessage, setAlertMessage] = useState("");
    const API_BASE_URL = 'http://localhost:8080/predict/image';
    const navigate = useNavigate();

    useEffect(() => {
        const handleDisplayResults = async () => {
            if (a === null) {
                if (!file) return;

                setLoading(true);
                const formData = new FormData();
                formData.append('image', file);

                try {
                    const response = await axios.post(API_BASE_URL, formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data',
                        },
                    });

                    const data = response.data;
                    if (data && typeof data === 'object' && 'prediction' in data && 'confidence' in data) {
                        const confidence = parseFloat(data.confidence.toFixed(2));
                        setResult(data.prediction);
                        setAccuracy(confidence);

                        if (confidence <= 60) {
                            setAlertMessage(TypeAlertMessageEnum.ErrorMessage);
                        } else if (confidence > 60 && confidence <= 70) {
                            setAlertMessage(TypeAlertMessageEnum.WarningMessage);
                        } else if (confidence > 70 && confidence <= 85) {
                            setAlertMessage(TypeAlertMessageEnum.InfoMessage);
                        } else {
                            setAlertMessage(TypeAlertMessageEnum.SuccessMessage);
                        }
                    } else {
                        console.error("Unexpected response structure:", data);
                    }
                } catch (error) {
                    console.error('Error uploading file:', error);
                } finally {
                    setLoading(false);
                }
            }
        };

        handleDisplayResults();
    }, [file]);

    const handleNavigateToInfo = () => {
        if (file && fileURL && result && accuracy) {
          navigate('/info', {
            state: { file, fileURL, result, accuracy },
          });
        }
      };

    return (
        <div className="page">
            {fileURL && <img src={fileURL} alt="Uploaded" style={{ maxWidth: "100%", maxHeight: "350px" }} />}
            <div className="card">
                {loading ? (
                    <MagnifyingGlass color="#65b2a0" />
                ) : (
                    <div>
                        <DisplayCard result={result} accuracy={accuracy} />
                        {alertMessage && (
                            <Alert severity={
                                alertMessage === TypeAlertMessageEnum.ErrorMessage
                                    ? 'error'
                                    : alertMessage === TypeAlertMessageEnum.WarningMessage
                                    ? 'warning'
                                    : alertMessage === TypeAlertMessageEnum.InfoMessage
                                    ? 'info'
                                    : 'success'
                            }>
                                {alertMessage}
                            </Alert>
                        )}
                    </div>
                )}
            </div>
            <div className="boutons">
                <Button
                    variant="contained"
                    onClick={() => navigate("/home")}
                    sx={{
                        backgroundColor: "#65b2a0",
                        color: "#fff",
                        "&:hover": { backgroundColor: "#386258" },
                    }}
                >
                    Analyze another image
                </Button>
                <Button
                    variant="contained"
                    startIcon={<InfoIcon />}
                    onClick={handleNavigateToInfo}
                    sx={{
                        backgroundColor: "#65b2a0",
                        color: "#fff",
                        "&:hover": { backgroundColor: "#386258" },
                    }}
                >
                    More info about the analysis
                </Button>
            </div>
        </div>
    );
};

export default ResultsPage;
