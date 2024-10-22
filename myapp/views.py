from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from myapp.serializers import *
from myapp.models import *
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
import numpy as np
from decimal import Decimal
import random
from django.core.mail import send_mail
from django.conf import settings
from rest_framework.decorators import permission_classes
from rest_framework import permissions
from rest_framework.permissions import AllowAny
from rest_framework.permissions import IsAuthenticated
import logging
import pickle
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

@api_view(['POST'])
def create_custom_user(request):
    if request.method == 'POST':
        serializer = CustomUserSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response({"status":"ok","message":"account created successfully","data":serializer.data}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

@api_view(['GET'])
def list_custom_users(request):
    users = CustomUser.objects.all()  
    serializer = CustomUserSerializer(users, many=True, context={'request': request}) 
    return Response({"status":"ok","message":"list users successfully","data":serializer.data}, status=status.HTTP_200_OK)


@api_view(['GET'])
def retrieve_custom_user(request, user_id):
    try:
        user = CustomUser.objects.get(id=user_id)  
    except CustomUser.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    
    serializer = CustomUserSerializer(user, context={'request': request})  
    return Response({"status":"ok","message":"user retrieved successfully","data":serializer.data}, status=status.HTTP_200_OK)



@api_view(['PUT'])
def update_custom_user(request, user_id):
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    serializer = CustomUserSerializer(user, data=request.data, partial=True, context={'request': request})
    if serializer.is_valid():
        serializer.save()
        return Response({"status": "ok", "message": "User updated successfully", "data": serializer.data}, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE'])
def delete_custom_user(request, user_id):
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    user.delete()
    return Response({"status": "ok", "message": "User deleted successfully"}, status=status.HTTP_204_NO_CONTENT)




@api_view(['POST'])
def login_user(request):
    username = request.data.get('username')
    password = request.data.get('password')
    
    user = authenticate(request, username=username, password=password)

    if user is not None:
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'name': user.name,
            }
        }, status=status.HTTP_200_OK)
    else:
        return Response({'error': 'Invalid email or password'}, status=status.HTTP_401_UNAUTHORIZED)
    
@api_view(['POST'])
def create_bank_account(request, user_id):
    if request.method == 'POST':
        # Attach the user_id to the incoming data
        data = request.data.copy()
        data['user_id'] = user_id

        serializer = BankSerializer(data=data)
        if serializer.is_valid():
            bank_account = serializer.save()  # Save the bank account and store the instance
            return Response({
                "status": "ok",
                "message": "Bank account added successfully",
                "data": {
                    "id": bank_account.id,  # Include the ID of the newly created bank account
                    **serializer.data  # Include the rest of the serialized data
                }
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def list_bank_accounts(request, user_id):
    if request.method == 'GET':
        # Retrieve all bank accounts associated with the user_id
        bank_accounts = BankAccount.objects.filter(user_id=user_id)
        serializer = BankSerializer(bank_accounts, many=True)
        return Response({
            "status": "ok",
            "message": "Bank accounts retrieved successfully",
            "data": serializer.data
        }, status=status.HTTP_200_OK)



@api_view(['PUT', 'PATCH'])
def update_bank_account(request, user_id, account_id):
    try:
        # Retrieve the specific bank account associated with the user_id and account_id
        bank_account = BankAccount.objects.get(user_id=user_id, id=account_id)
    except BankAccount.DoesNotExist:
        return Response({"status": "error", "message": "Bank account not found"}, status=status.HTTP_404_NOT_FOUND)

    # Partial updates are allowed with PATCH; full updates are expected with PUT
    serializer = BankSerializer(bank_account, data=request.data, partial=(request.method == 'PATCH'))

    if serializer.is_valid():
        serializer.save()
        return Response({
            "status": "ok",
            "message": "Bank account updated successfully",
            "data": serializer.data
        }, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@api_view(['GET'])
def bank_account_detail(request, user_id, account_id):
    try:
        # Retrieve the specific bank account associated with the user_id and account_id
        bank_account = BankAccount.objects.get(user_id=user_id, id=account_id)
    except BankAccount.DoesNotExist:
        return Response({"status": "error", "message": "Bank account not found"}, status=status.HTTP_404_NOT_FOUND)

    # Serialize the bank account details
    serializer = BankSerializer(bank_account)
    return Response({
        "status": "ok",
        "message": "Bank account retrieved successfully",
        "data": serializer.data
    }, status=status.HTTP_200_OK)





logger = logging.getLogger(__name__)

# Load the model, scaler, and PCA
def load_model():
    with open('fraud_detection_model.pkl', 'rb') as f:
        return pickle.load(f)

# Function to predict fraud
def predict_fraud(time_hours, amount):
    saved_objects = load_model()
    scaler = saved_objects['scaler']
    pca = saved_objects['pca']
    model = saved_objects['model']

    # Prepare input features
    input_data = pd.DataFrame({
        'Time': [time_hours * 3600],  # Convert hours to seconds
        'Amount': [amount]
    })

    # Standardize the input data
    input_scaled = scaler.transform(input_data)

    # Apply PCA
    input_pca = pca.transform(input_scaled)

    # Make a prediction
    prediction = model.predict(input_pca)

    return prediction[0]  # Return the prediction (0: Legit, 1: Fraud)

@csrf_exempt
def create_transaction(request, user_id):
    if request.method == 'POST':
        try:
            # Get transaction details from the request
            amount_str = request.POST.get('amount')  # Get amount as string
            ac_number = request.POST.get('ac_number')
            ifsc_code = request.POST.get('ifsc_code')
            nameOrig = request.POST.get('nameOrig')
            transaction_type = request.POST.get('transaction_type')

            # Check if amount is provided
            if amount_str is None:
                return JsonResponse({'error': 'Amount is required.'}, status=400)

            # Convert to float
            amount = float(amount_str)

            # Get the current time automatically
            current_time = timezone.now().time()  # Get the current time

            # Predict fraud
            prediction = predict_fraud(current_time.hour + current_time.minute / 60.0, amount)

            # Get the user's email using user_id
            User = get_user_model()  # Get the custom user model
            user = User.objects.get(id=user_id)  # Fetch the user based on user_id
            user_email = user.email  # Get the user's email

            # If fraud is detected (prediction == 1), send email and prevent saving transaction
            if prediction == 1:
                # Send fraud alert email to the user
                send_mail(
                    subject='Fraudulent Transaction Alert',
                    message=f'Dear {nameOrig},\n\nA fraudulent transaction was detected in your account.\n'
                            f'Details:\n'
                            f'Amount: {amount}\n'
                            f'Account Number: {ac_number}\n'
                            f'IFSC Code: {ifsc_code}\n'
                            f'Transaction Type: {transaction_type}\n\n'
                            f'Please review the transaction immediately.',
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user_email],
                    fail_silently=False,
                )

                # Log fraud detection and return failure response
                logger.warning(f"Transaction failed due to fraud detection for user {nameOrig}, amount: {amount}")
                
                # Return JSON response indicating fraud detected and transaction failed
                return JsonResponse({
                    'message': 'Transaction failed: Fraud detected',
                    'is_fraud': 1,
                }, status=403)

            # If no fraud detected, save the transaction
            transaction = Transaction.objects.create(
                amount=amount,
                ac_number=ac_number,
                ifsc_code=ifsc_code,
                nameOrig=nameOrig,
                transaction_type=transaction_type,
                is_fraud=bool(prediction),  # Save fraud status
                user_id=user_id,
                time=current_time  # Set the time automatically
            )

            # Return successful transaction details as a JSON response
            response_data = {
                'is_fraud': 0,
                'transaction_id': transaction.id,
                'amount': transaction.amount,
                'ac_number': transaction.ac_number,
                'ifsc_code': transaction.ifsc_code,
                'nameOrig': transaction.nameOrig,
                'transaction_type': transaction.transaction_type,
                'time': str(transaction.time)  # Convert time to string for JSON serialization
            }

            return JsonResponse(response_data)

        except ValueError as ve:
            logger.error(f"Value error during fraud prediction or transaction creation: {str(ve)}")
            return JsonResponse({'error': str(ve)}, status=400)
        except Exception as e:
            logger.error(f"Error during fraud prediction or transaction creation: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'message': 'Invalid request method'}, status=405)
@api_view(['GET'])
def list_transactions(request, user_id):
    # Filter transactions by user_id
    transactions = Transaction.objects.filter(user_id=user_id)

    # Serialize the transaction data
    serializer = TransactionSerializer(transactions, many=True)

    return Response({
        'status': 'ok',
        'message':'transaction history',
        'data': serializer.data
    }, status=status.HTTP_200_OK)






ADVICE_LIST = [
    {"id": 1, "category": "security", "text": "Always use two-factor authentication."},
    {"id": 2, "category": "transaction", "text": "Double-check the recipient's details before sending funds."},
    {"id": 3, "category": "privacy", "text": "Avoid using public Wi-Fi for sensitive transactions."},
    {"id": 4, "category": "awareness", "text": "Be cautious of phishing emails."},
    {"id": 5, "category": "security", "text": "Keep your software updated to protect against vulnerabilities."},
    {"id": 6, "category": "finance", "text": "Create a budget and stick to it."},
    {"id": 7, "category": "investment", "text": "Diversify your investments to reduce risk."},
    {"id": 8, "category": "planning", "text": "Always have an emergency fund for unexpected expenses."},
    {"id": 9, "category": "spending", "text": "Limit impulse purchases to stay within budget."},
    {"id": 10, "category": "savings", "text": "Aim to save at least 20% of your income."},
    {"id": 11, "category": "education", "text": "Keep learning about personal finance."},
    {"id": 12, "category": "goal-setting", "text": "Set clear financial goals to stay focused."},
    {"id": 13, "category": "debt", "text": "Pay off high-interest debt as soon as possible."},
    {"id": 14, "category": "shopping", "text": "Use a shopping list to avoid unnecessary purchases."},
    {"id": 15, "category": "financial literacy", "text": "Read books about personal finance and investing."},
]

@api_view(['GET'])
def random_advice_view(request):
    random_advice = random.choice(ADVICE_LIST)  
    return Response(random_advice)


@api_view(['POST'])
def password_reset_view(request):
    serializer = PasswordResetSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()  # No arguments passed here
        return Response({"detail": "Password reset successful."}, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)