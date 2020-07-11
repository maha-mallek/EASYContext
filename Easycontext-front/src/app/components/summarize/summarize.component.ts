import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, FormControl, Validators } from '@angular/forms';
import { UserService } from 'src/app/services/user.service';
import { Router } from '@angular/router';

import { ToastrService } from 'ngx-toastr';
@Component({
  selector: 'app-summarize',
  templateUrl: './summarize.component.html',
  styleUrls: ['./summarize.component.css']
})
export class SummarizeComponent implements OnInit {
 summary;
 load:boolean;
 res:boolean;
 TextForm: FormGroup;
 fileForm:FormGroup;
 constructor(private fb: FormBuilder,private summarizedata:UserService,private router : Router,private toastr: ToastrService) { 
   let formControls = {
   texte: new FormControl('', [
     Validators.required,
     Validators.maxLength(30000),
   ])
 }
 let formControls2 = {
  file: new FormControl('', [
    //Validators.required,
    //Validators.maxLength(15000),
  ])
  }

 this.TextForm = fb.group(formControls);
 this.fileForm=fb.group(formControls2);
}

get texte() { return this.TextForm.get('texte'); }

 ngOnInit(): void {
   this.summary = {
     text: '',
     summarize:'',
   };
   this.load=false;
   this.res=false;
 }
posttext(){
  this.load=true;
  this.summarizedata.posttext(this.summary).subscribe(
   result => {
    this.load=false;
    this.res=true;
     console.log(result);
     //alert("summarized successfully !!");     
     var json = JSON.parse(JSON.stringify(result[0]));
     this.toastr.success('Your Text Is Summarized Successfully !');/*success/error/warning/info/show()*/

     this.summary = {
       text: json["text"],
       summarize: json["summarize"],
      
     };
     
   },
   error => {
     console.log(error);
     this.toastr.error('An error occured in our servers please retry !');/*success/error/warning/info/show()*/
     this.load=false;
   }
 );
 
 
}

  public onChange(fileList: FileList): void {
    let file = fileList[0];
    let fileReader: FileReader = new FileReader();
    let self = this;
    fileReader.onloadend = function(x) {
      self.summary.text = fileReader.result as string;
    }
    fileReader.readAsText(file);
  }

New(){
 
  this.router.navigate(['summarize']); 
}
}