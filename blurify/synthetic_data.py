"""
Synthetic data generator for PII replacement.

Generates realistic random data to replace detected PII fields instead of blurring.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import re


class SyntheticDataGenerator:
    """Generates realistic synthetic data to replace PII."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducible results."""
        if seed:
            random.seed(seed)
        
        # Track used replacements to avoid duplicates in same document
        self._used_names: Set[str] = set()
        self._used_emails: Set[str] = set()
        self._used_phones: Set[str] = set()
        
        # Sample data pools
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
            "James", "Mary", "William", "Jennifer", "Richard", "Patricia", "Charles",
            "Linda", "Joseph", "Elizabeth", "Thomas", "Barbara", "Christopher", "Susan",
            "Daniel", "Jessica", "Matthew", "Nancy", "Anthony", "Dorothy", "Mark",
            "Ashley", "Donald", "Helen", "Steven", "Kimberly", "Paul", "Donna",
            "Andrew", "Carol", "Joshua", "Ruth", "Kenneth", "Sharon", "Kevin", "Michelle",
            "Brian", "Laura", "George", "Sarah", "Edward", "Kimberly", "Ronald", "Deborah"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
            "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
            "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
            "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell"
        ]
        
        self.email_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
            "icloud.com", "protonmail.com", "mail.com", "zoho.com", "fastmail.com"
        ]
        
        # Indian specific data for local context
        self.indian_first_names = [
            "Arjun", "Priya", "Rahul", "Sneha", "Vikram", "Pooja", "Amit", "Kavya",
            "Ravi", "Meera", "Suresh", "Divya", "Karan", "Neha", "Rajesh", "Swati",
            "Anil", "Shweta", "Deepak", "Asha", "Manoj", "Rekha", "Vinod", "Sunita"
        ]
        
        self.indian_last_names = [
            "Sharma", "Gupta", "Singh", "Kumar", "Patel", "Agarwal", "Jain", "Shah",
            "Verma", "Mehta", "Chopra", "Malhotra", "Bansal", "Aggarwal", "Bhatia"
        ]

    def generate_name(self, original_name: str, prefer_indian: bool = True) -> str:
        """Generate a random name replacement."""
        # Determine if we should use Indian or Western names
        first_pool = self.indian_first_names if prefer_indian else self.first_names
        last_pool = self.indian_last_names if prefer_indian else self.last_names
        
        # Try to match the structure of the original name
        name_parts = original_name.strip().split()
        
        if len(name_parts) == 1:
            # Single name - generate first name only
            replacement = random.choice(first_pool)
        elif len(name_parts) == 2:
            # First and last name
            first = random.choice(first_pool)
            last = random.choice(last_pool)
            replacement = f"{first} {last}"
        else:
            # Multiple names - generate first, middle (optional), last
            first = random.choice(first_pool)
            last = random.choice(last_pool)
            if len(name_parts) > 2:
                middle_initial = random.choice(string.ascii_uppercase)
                replacement = f"{first} {middle_initial}. {last}"
            else:
                replacement = f"{first} {last}"
        
        # Ensure uniqueness within document
        attempts = 0
        while replacement in self._used_names and attempts < 10:
            first = random.choice(first_pool)
            last = random.choice(last_pool)
            replacement = f"{first} {last}"
            attempts += 1
        
        self._used_names.add(replacement)
        return replacement

    def generate_email(self, original_email: str) -> str:
        """Generate a random email replacement."""
        # Extract domain to potentially preserve organization context
        original_domain = original_email.split('@')[-1] if '@' in original_email else None
        
        # Generate username (6-12 characters)
        username_length = random.randint(6, 12)
        username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
        
        # Choose domain (prefer generic ones for privacy)
        domain = random.choice(self.email_domains)
        
        replacement = f"{username}@{domain}"
        
        # Ensure uniqueness
        attempts = 0
        while replacement in self._used_emails and attempts < 10:
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
            replacement = f"{username}@{domain}"
            attempts += 1
        
        self._used_emails.add(replacement)
        return replacement

    def generate_phone(self, original_phone: str, prefer_indian: bool = True) -> str:
        """Generate a random phone number replacement that is GUARANTEED to be different."""
        # Clean original to understand format
        digits_only = re.sub(r'[^\d]', '', original_phone)
        original_digits = digits_only
        
        # Generate replacement based on format
        if prefer_indian and len(digits_only) == 10:
            # Indian mobile number format - ensure it's different
            new_number = self._generate_different_indian_number(original_digits)
            
            # Try to match original formatting
            if '+91' in original_phone:
                replacement = f"+91 {new_number[:5]} {new_number[5:]}"
            elif original_phone.count(' ') >= 1:
                replacement = f"{new_number[:5]} {new_number[5:]}"
            elif original_phone.count('-') >= 1:
                replacement = f"{new_number[:5]}-{new_number[5:]}"
            else:
                replacement = new_number
                
        elif prefer_indian and len(digits_only) == 12 and digits_only.startswith('91'):
            # Indian number with country code (+91 XXXXXXXXXX)
            new_number = self._generate_different_indian_number(digits_only[2:])  # Remove 91 prefix
            
            # Format with country code
            if '+91' in original_phone:
                replacement = f"+91 {new_number[:5]} {new_number[5:]}"
            elif '91' in original_phone:
                replacement = f"91 {new_number[:5]} {new_number[5:]}"
            else:
                replacement = f"91{new_number}"
                
        elif len(digits_only) == 11 and digits_only.startswith('1'):
            # US/Canada format
            area_code = random.randint(200, 999)
            exchange = random.randint(200, 999)
            number = random.randint(1000, 9999)
            
            if '(' in original_phone and ')' in original_phone:
                replacement = f"1 ({area_code}) {exchange}-{number}"
            elif '-' in original_phone:
                replacement = f"1-{area_code}-{exchange}-{number}"
            else:
                replacement = f"1{area_code}{exchange}{number}"
                
        else:
            # Generic format - maintain length and ensure different
            max_attempts = 20
            for attempt in range(max_attempts):
                replacement = ''.join([str(random.randint(0, 9)) for _ in range(len(digits_only))])
                if replacement != digits_only:
                    break
        
        # Ensure uniqueness
        attempts = 0
        while replacement in self._used_phones and attempts < 10:
            if prefer_indian and len(digits_only) == 10:
                new_number = self._generate_different_indian_number(original_digits)
                replacement = new_number
            else:
                replacement = ''.join([str(random.randint(0, 9)) for _ in range(len(digits_only))])
            attempts += 1
        
        self._used_phones.add(replacement)
        return replacement

    def _generate_different_indian_number(self, original_number: str) -> str:
        """Generate a different Indian phone number ensuring uniqueness."""
        max_attempts = 50
        for attempt in range(max_attempts):
            # Generate a new 10-digit number
            first_digit = random.choice([7, 8, 9])  # Valid Indian mobile first digits
            remaining_digits = [random.randint(0, 9) for _ in range(9)]
            new_number = str(first_digit) + ''.join(map(str, remaining_digits))
            
            # Ensure it's different from original
            if new_number != original_number and new_number not in self._used_phones:
                return new_number
                
        # Fallback if all attempts failed
        return f"{random.choice([7, 8, 9])}{random.randint(100000000, 999999999)}"

    def generate_date(self, original_date: str) -> str:
        """Generate a random date replacement maintaining format."""
        try:
            # Common date patterns
            patterns = [
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%d/%m/%Y'),  # DD/MM/YYYY
                (r'(\d{1,2})-(\d{1,2})-(\d{4})', '%d-%m-%Y'),  # DD-MM-YYYY
                (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),  # YYYY-MM-DD
                (r'(\d{1,2}) (\w+) (\d{4})', '%d %B %Y'),       # DD Month YYYY
            ]
            
            for pattern, date_format in patterns:
                if re.match(pattern, original_date.strip()):
                    # Generate random date in past 30 years
                    start_date = datetime.now() - timedelta(days=365*30)
                    end_date = datetime.now() - timedelta(days=365*18)  # At least 18 years old
                    
                    random_date = start_date + timedelta(
                        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
                    )
                    
                    # Format according to original pattern
                    if '%B' in date_format:  # Full month name
                        return random_date.strftime(date_format)
                    else:
                        return random_date.strftime(date_format)
            
            # Fallback - return original if can't parse
            return original_date
            
        except Exception:
            return original_date

    def generate_aadhaar(self, original_aadhaar: str) -> str:
        """Generate a random Aadhaar number replacement."""
        # Generate 12 random digits (avoiding real Aadhaar patterns)
        digits = ''.join([str(random.randint(0, 9)) for _ in range(12)])
        
        # Match original formatting
        if ' ' in original_aadhaar:
            return f"{digits[:4]} {digits[4:8]} {digits[8:]}"
        elif '-' in original_aadhaar:
            return f"{digits[:4]}-{digits[4:8]}-{digits[8:]}"
        else:
            return digits

    def generate_pan(self, original_pan: str) -> str:
        """Generate a random PAN number replacement."""
        # PAN format: ABCDE1234F (5 letters, 4 digits, 1 letter)
        letters1 = ''.join(random.choices(string.ascii_uppercase, k=5))
        digits = ''.join(random.choices(string.digits, k=4))
        letter2 = random.choice(string.ascii_uppercase)
        
        return f"{letters1}{digits}{letter2}"

    def generate_replacement(self, pii_type: str, original_text: str, prefer_indian: bool = True) -> str:
        """Generate appropriate replacement based on PII type."""
        pii_type = pii_type.lower()
        
        if pii_type in ['person_name', 'name', 'person']:
            return self.generate_name(original_text, prefer_indian)
        elif pii_type in ['email', 'email_address']:
            return self.generate_email(original_text)
        elif pii_type in ['phone', 'phone_number', 'mobile']:
            return self.generate_phone(original_text, prefer_indian)
        elif pii_type in ['date', 'birth_date', 'dob']:
            return self.generate_date(original_text)
        elif pii_type in ['aadhaar', 'aadhaar_number']:
            return self.generate_aadhaar(original_text)
        elif pii_type in ['pan', 'pan_number']:
            return self.generate_pan(original_text)
        else:
            # Generic replacement - replace with random alphanumeric
            return ''.join(random.choices(string.ascii_letters + string.digits, k=len(original_text)))

    def reset_session(self):
        """Reset used data tracking for new document."""
        self._used_names.clear()
        self._used_emails.clear()
        self._used_phones.clear()


# Example usage and testing
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
    # Test name generation
    print("Name replacements:")
    print(f"Chirag Agrawal -> {generator.generate_name('Chirag Agrawal')}")
    print(f"Milan Sharma -> {generator.generate_name('Milan Sharma')}")
    print(f"Nancy -> {generator.generate_name('Nancy')}")
    
    # Test email generation
    print("\nEmail replacements:")
    print(f"chirag@example.com -> {generator.generate_email('chirag@example.com')}")
    print(f"milan.sharma@company.org -> {generator.generate_email('milan.sharma@company.org')}")
    
    # Test phone generation
    print("\nPhone replacements:")
    print(f"9876543210 -> {generator.generate_phone('9876543210')}")
    print(f"+91 98765 43210 -> {generator.generate_phone('+91 98765 43210')}")
    
    # Test date generation
    print("\nDate replacements:")
    print(f"15/03/1990 -> {generator.generate_date('15/03/1990')}")
    print(f"25 Jan 2023 -> {generator.generate_date('25 Jan 2023')}")
    
    # Test Aadhaar/PAN
    print("\nID replacements:")
    print(f"1234 5678 9012 -> {generator.generate_aadhaar('1234 5678 9012')}")
    print(f"ABCDE1234F -> {generator.generate_pan('ABCDE1234F')}")