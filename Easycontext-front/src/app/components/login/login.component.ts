import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, FormBuilder, Validators } from '@angular/forms';
import { UserService } from 'src/app/services/user.service';
import { JwtHelperService, JwtModule } from '@auth0/angular-jwt';
import { Router } from '@angular/router';
import { User } from 'src/app/models/user';
import { AuthService } from 'src/app/services/auth.service';

import { ToastrService } from 'ngx-toastr';
@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
  isAuth : Boolean;
  loginForm: FormGroup;
  loginn;
  constructor(private fb: FormBuilder,  private _as: AuthService, private router: Router,private toastr: ToastrService) {
    let token = localStorage.getItem('token');
    /*if (token) {
      if (_as.isAdmin()) {
        router.navigate(['/admin'])
      } else if (_as.isStudent()) {
        const helper = new JwtHelperService();
        const studentId = helper.decodeToken(token).studentId;
        router.navigate(['/contextextraction', studentId])
      }
    }*/
    let formControls = {
      username: new FormControl('',[
        Validators.required,
        Validators.minLength(4)
      ]),
      password: new FormControl('',[
        Validators.required,
        Validators.minLength(4)
      ])
    }

    this.loginForm = fb.group(formControls);
  }

  //get email(){return this.loginForm.get('email');}
  get username(){return this.loginForm.get('username');}
  get password(){return this.loginForm.get('password');}


  ngOnInit(): void {
    this.loginn = {
      username: '',
      password: '',
    };
  }

  login() {
    //let data = this.loginForm.value;
    //let user = new User(data.username, null, data.password);

    this._as.loginUser(this.loginn).subscribe(
     
      result => {
        //alert('User '+ this.loginn.username + ' has been logged successfully!');
        //console.log(result);
        //console.log(data);
        this.toastr.success('You Have Logged IN Successfully!');/*success/error/warning/info/show()*/
        let token = result['token'];
        //this.router.navigateByUrl('/');
        localStorage.setItem('token', token);
        this._as.isAuthenticated();  
        const helper = new JwtHelperService();

        const decodedToken = helper.decodeToken(token);
        if (decodedToken.user_id == 1){
          window.location.href = 'http://localhost:8000/admin/';
        }
        else{
          window.location.href = 'http://localhost:4200/';
        }
      },
      error => {
        this.toastr.warning('Wrong credentials , Please retry !');
        //alert('Wrong credentials , Please retry!');
        this.router.navigate(['/login']);
        console.log(error);
      }
    );

  }

}
