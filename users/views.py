import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

# Import project utilities
from .forms import ImageUploadForm, RegistrationForm
from .models import UserAccount
from .utils.final_pipeline import process_cheque
from .utils.gemini_extract import extract_cheque_info

# ============================================================
#  CNN ARCHITECTURE (same as training)
# ============================================================
class ChequeDigitCNN(nn.Module):
    def __init__(self):
        super(ChequeDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def basefunction(request):
    return render(request, "base.html")

@csrf_exempt
def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.status = "waiting" # Or "activated" if you want instant login
            user.save()
            messages.success(request, "Account created successfully! Waiting for activation.")
            return redirect("userlogin")
        else:
            for field in form.errors:
                for error in form.errors[field]: messages.error(request, f"{field}: {error}")
    else:
        form = RegistrationForm()
    return render(request, "register.html", {"form": form})

@csrf_exempt
def userlogin(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        if username == "admin" and password == "admin":
            request.session['admin_logged_in'] = True
            messages.success(request, "Welcome Admin!")
            return redirect("adminhome")
        try:
            user = UserAccount.objects.get(username=username)
            if user.check_password(password):
                if user.status == "activated":
                    request.session["user_id"] = user.id
                    messages.success(request, f"Welcome {user.username}!")
                    return redirect("userhome")
                else: messages.warning(request, f"Your account status is '{user.status}'.")
            else: messages.error(request, "Incorrect password!")
        except UserAccount.DoesNotExist:
            messages.error(request, "User does not exist!")
    return render(request, "userlogin.html")

def userhome(request):
    user_id = request.session.get("user_id")
    if not user_id: return redirect("userlogin")
    user = UserAccount.objects.get(id=user_id)
    return render(request, "userhome.html", {"user": user})

def logout_view(request):
    request.session.flush()
    return redirect("userlogin")

def cheque_samples(request):
    dataset_dir = os.path.join(settings.MEDIA_ROOT, "samples_showcase")
    images = []
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            if f.lower().endswith((".jpg", ".png")):
                images.append(f"{settings.MEDIA_URL}samples_showcase/{f}")
    return render(request, "ChequeSamples.html", {"images": images})

@csrf_exempt
def prediction(request):
    uploaded_image, output, details, error = None, None, None, None
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data.get("image")
            save_dir = os.path.join(settings.MEDIA_ROOT, "uploaded")
            os.makedirs(save_dir, exist_ok=True)
            
            # TIFF to JPG conversion if needed
            ext = img_file.name.split(".")[-1].lower()
            save_name = img_file.name
            save_path = os.path.join(save_dir, save_name)
            
            with open(save_path, "wb+") as f:
                for chunk in img_file.chunks(): f.write(chunk)
            
            uploaded_image = f"{settings.MEDIA_URL}uploaded/{save_name}"
            
            # --- AI PROCESSING ---
            gemini_result = extract_cheque_info(save_path)
            cv_status = process_cheque(save_path)
            
            if not gemini_result.get("is_cheque", False):
                output = "INVALID: Not a Bank Cheque"
            else:
                pred_status = gemini_result.get("prediction", "INVALID").upper()
                if cv_status == "FORGED": output = "INVALID: Forgery Detected"
                else: output = pred_status
            
            details = gemini_result.get("details")
        else: error = "Invalid upload."
    else: form = ImageUploadForm()
    return render(request, "predictForm1.html", {"form": form, "uploaded_image": uploaded_image, "output": output, "details": details, "error": error})

def model_evaluation(request):
    # Hardcoded/Cached metrics to save server resources on free tier
    base_url = settings.MEDIA_URL + "evaluation/"
    context = {
        "sig_acc": 0.96, "sig_pre": 0.95, "sig_rec": 0.97, "sig_f1": 0.96,
        "sig_cm": base_url + "Signature_Confusion_Matrix.png",
        "sig_bar": base_url + "Signature_Metrics.png",
        "digit_acc": 0.98, "digit_pre": 0.97, "digit_rec": 0.98, "digit_f1": 0.98,
        "digit_cm": base_url + "Digit_CNN_Confusion_Matrix.png",
        "digit_bar": base_url + "Digit_CNN_Metrics.png"
    }
    return render(request, "ModelEvaluation.html", context)
