import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, FormControl, Validators } from '@angular/forms';
import { JwtHelperService } from '@auth0/angular-jwt';
import { UserService } from 'src/app/services/user.service';
import { Router } from '@angular/router';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-admin',
  templateUrl: './admin.component.html',
  styleUrls: ['./admin.component.css']
})
export class AdminComponent implements OnInit {
  UserForm: FormGroup;
  user;
  user_u;
  id;
  constructor(private fb: FormBuilder , private userService : UserService, private router : Router,private toastr: ToastrService) { 
    let formControls = {
      username: new FormControl('',[
        Validators.required,
        Validators.minLength(4)
        
      ]),
      email: new FormControl('',[
        Validators.required,
        Validators.email
      ])
    }

    this.UserForm = fb.group(formControls);
  }
  
  get email(){return this.UserForm.get('email');}
  get username(){return this.UserForm.get('username');}

  ngOnInit(): void {
    let token= localStorage.getItem('token');
  
  const helper = new JwtHelperService();
  
  const decodedToken = helper.decodeToken(token);
  this.id=decodedToken.user_id;
  this.user={
    name:decodedToken.username,
    mail:decodedToken.email,
  }
  this.user_u={
    username:'',
    email:''
  }
  
  }
  
  update(){
      this.userService.update_informations(this.id,this.user_u)
      .subscribe(
        response => {
          console.log(response);
          this.toastr.success('Your personal Informations Are Now Updated Successfully ! You Have to login Now !');/*success/error/warning/info/show()*/
          this.router.navigateByUrl('login');
          console.log(this.user);
  
        },
        error => {
          console.log('error',error);
          this.toastr.warning('This username is used ! Try to change it !');/*success/error/warning/info/show()*/
        }
        );
    }
}
