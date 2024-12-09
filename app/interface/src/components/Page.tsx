import { FileUploader } from "react-drag-drop-files";
import "./Page.css";
import { useState } from "react";
import { Button } from "@mui/material";
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import axios from 'axios';

const fileTypes = ["JPG"];

const Page = () => {
  const [file, setFile] = useState<File | null>(null);
  const [disableButton, setDisableButton] = useState(true);
  const [fileURL, setFileURL] = useState<string | null>(null); // Pour stocker l'URL de l'image
  const [displayDragAndDropPage,setDisplayDragAndDropPage] = useState("DRAGANDDROP");
  const API_BASE_URL = 'http://localhost:8080/predict/image';

  const handleChange = (file: File) => {
    setFile(file);
    setDisableButton(file == null);

    // Générer l'URL de l'image et l'enregistrer
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setFileURL(imageUrl);
    }
  };

  const handleDisplayDragAndDropPage = (value:string) => {
    setDisplayDragAndDropPage(value);
    if (value === "DRAGANDDROP") {
      setDisableButton(true);
      setFileURL(null); // Réinitialiser l'URL si nécessaire
      setFile(null);
    } else {
      try {
        // On envoie l'image au modèle
        if (file) {
          const formData = new FormData();
          formData.append('file', file);
  
          axios.post(
            API_BASE_URL,
            formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          }
        );
      }
      } catch (error) {
        console.error('Error uploading file:', error);
      }

    }
  }
  // TODO : Obtenir le résultat du modèle : Result + accuracy
  const result = "NOT GENERATED BY AN AI";
  //const result = "GENERATED BY AN AI"
  const accuracy = 92


  return (
    <div className="page">
      {displayDragAndDropPage === "DRAGANDDROP" ? (
        <div>
          <div className="page">
            <h3>Drag & Drop your image here</h3>
            <FileUploader
              multiple={false}
              handleChange={handleChange}
              name="file"
              types={fileTypes}
            />
            <p>{file != null ? `File name: ${file.name}` : "No file uploaded yet"}</p>
            {fileURL && <img src={fileURL} alt="Preview" style={{ maxWidth: "100%", maxHeight: "300px" }} />}
          </div>
          <Button
            disabled={disableButton}
            variant="contained"
            onClick={() => handleDisplayDragAndDropPage("RESULTS")}
            sx={{
              marginTop:"20px",
              backgroundColor: "#65b2a0",
              color: "#fff",
              "&:hover": { backgroundColor: "#386258" },
            }}
          >
            Go
          </Button>
        </div>
      ) : (
        <div className="page">
          <h3> TODO: Display a <CheckIcon/> or a <CloseIcon/></h3>
          {fileURL && <img src={fileURL} alt="Uploaded" style={{ maxWidth: "100%", maxHeight: "350px"}} />}
          <h2> After an analysis, we think this image is {result}. </h2>
          <h3> The accuracy of the result is {accuracy}% </h3>
          <Button
          variant="contained"
          onClick={() => handleDisplayDragAndDropPage("DRAGANDDROP")}
          sx={{
            backgroundColor: "#65b2a0",
            color: "#fff",
            "&:hover": { backgroundColor: "#386258" },
          }}
          >
            Analyze an other image
          </Button>
        </div>
      )}
    </div>
  );
};

export default Page;
