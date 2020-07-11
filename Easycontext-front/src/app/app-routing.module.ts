import { NgModule, Component } from '@angular/core';
import { Routes, RouterModule   } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { LoginComponent } from './components/login/login.component';
import { RegisterComponent } from './components/register/register.component';
import { SummarizeComponent } from './components/summarize/summarize.component';
import { DownloadComponent } from './components/download/download.component';
import { ContextExtractComponent } from './components/context-extract/context-extract.component';
import { AdminComponent } from './components/admin/admin.component';


import { AuthGuardService as AuthGuard } from './services/auth-guard.service';
import { RoleGuardService as RoleGuard } from './services/role-guard-service.service';

const routes: Routes = [
  {
    path:'',
    component:HomeComponent
  },
  {
    path :'profile',
    component : AdminComponent ,
    canActivate : [AuthGuard]
    
  },
  {
    path : 'login',
    component: LoginComponent,
   // canActivate : [AuthGuard]
  },  
  {
    path : 'register',
    component:RegisterComponent,
  },
  {
    path : 'summarize',
    component:SummarizeComponent,
    canActivate : [AuthGuard]
  },
  {
    path : 'download',
    component:DownloadComponent,
    canActivate : [AuthGuard]
  },
  {
    path :'contextextraction',
    component : ContextExtractComponent,
    canActivate : [AuthGuard]
  },
  {
    path : '**',
    component : LoginComponent
  },
  {
    path : 'logout', redirectTo : 'login' 
  },
  
  
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
