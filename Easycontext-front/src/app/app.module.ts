import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { HttpClientModule, HttpClientJsonpModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { LoginComponent } from './components/login/login.component';
import { HomeComponent } from './components/home/home.component';
import { HeaderComponent } from './components/header/header.component';
import { RegisterComponent } from './components/register/register.component';
import { SummarizeComponent } from './components/summarize/summarize.component';
import { DownloadComponent } from './components/download/download.component';
import { SidebarComponent } from './components/sidebar/sidebar.component';
import { ContextExtractComponent } from './components/context-extract/context-extract.component';
import { FooterComponent } from './components/footer/footer.component';
import { AdminComponent } from './components/admin/admin.component';

import { AuthGuardService } from './services/auth-guard.service';
import {
  ToastrModule,
  ToastNoAnimation,
  ToastNoAnimationModule
} from 'ngx-toastr';

@NgModule({
  declarations: [
    AppComponent,
    LoginComponent,
    HomeComponent,
    HeaderComponent,
    RegisterComponent,
    SummarizeComponent,
    DownloadComponent,
    SidebarComponent,
    ContextExtractComponent,
    FooterComponent,
    AdminComponent,
   
    

  ],
  imports: [
    BrowserModule,
    HttpClientJsonpModule,
    AppRoutingModule,
    ReactiveFormsModule,
    FormsModule,
    HttpClientModule,
    ToastNoAnimationModule.forRoot(),
  ],
  providers: [AuthGuardService],
  bootstrap: [AppComponent]
})
export class AppModule { }
