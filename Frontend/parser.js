document.addEventListener('DOMContentLoaded',()=>{const loginForm = document.getElementById('loginForm');


    loginForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const usernameInput = document.getElementById('username').value;
        const passwordInput = document.getElementById('password').value;
        const payload = {
            username: usernameInput,
            password: passwordInput
        };
        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Login failed');
            }

            const data = await response.json();
            
            if (data && data.access_token) {
                console.log('Login Successful!', data);
              
                localStorage.setItem('token', data.access_token);
   
            } else {
                alert('Invalid credentials');
            }

        } catch (error) {
            console.error('Error during login:', error);
            alert('Something went wrong. Please try again.');
        }
    });
});