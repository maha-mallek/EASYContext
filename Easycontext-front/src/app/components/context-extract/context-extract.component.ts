import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, Validators, FormControl } from '@angular/forms';
import { UserService } from 'src/app/services/user.service';
import { Router } from '@angular/router';
import { JwtHelperService } from '@auth0/angular-jwt';


import { ToastrService } from 'ngx-toastr';
@Component({
  selector: 'app-context-extract',
  templateUrl: './context-extract.component.html',
  styleUrls: ['./context-extract.component.css']
})
export class ContextExtractComponent implements OnInit {
  context;
  txt :String;
  load:boolean;
  res:boolean;
  TextForm: FormGroup;
  fileForm:FormGroup;
  constructor(private fb: FormBuilder,private extractdata:UserService,private router : Router,private toastr: ToastrService) { 
    let formControls = {
    texte: new FormControl('', [
      Validators.required,
      Validators.maxLength(30000),
    ])
  }
  let formControls2 = {
   file: new FormControl('', [
     //Validators.required,
     Validators.maxLength(30000),
   ])
   }
 
   this.TextForm = fb.group(formControls);
   this.fileForm = fb.group(formControls2);
  }

  get texte() { return this.TextForm.get('texte'); }
  get file() { return this.fileForm.get('file');}
  ngOnInit(): void {
    this.context= {
      text: '',
      user_id:'',
      contexte:'',
      keyword0:'',
      keyword1:'',
      keyword2:'',
      keyword3:'',
      keyword4:'',
      keyword5:'',
      keyword6:'',
      keyword7:'',
      keyword8:'',
      keyword9:'',
    };
    this.load=false;
    this.res=false;
  }
  Contextextraction(){
    this.load=true;
    
    this.txt=this.context['text']
    let token = localStorage.getItem('token');
    const helper = new JwtHelperService();

    const decodedToken = helper.decodeToken(token);
    this.context['id_user']=decodedToken.user_id 
    console.log(this.context['id_user'])
    this.extractdata.Contextextraction(this.context).subscribe(
     result => {
      this.load=false;
      this.res=true;
      
       var json = JSON.parse(JSON.stringify(result));//[0]
       //console.log(json['context'][0]['etiquette'])
       //console.log(json['keywords'][0]['mots'])
       this.context = {
        text: this.txt,
        contexte: json['context']['etiquette'],
        keyword0: json['keywords'][0]['mots'],
        keyword1: json['keywords'][1]['mots'],
        keyword2: json['keywords'][2]['mots'],
        keyword3: json['keywords'][3]['mots'],
        keyword4: json['keywords'][4]['mots'],
        keyword5: json['keywords'][5]['mots'],
        keyword6: json['keywords'][6]['mots'],
        keyword7: json['keywords'][7]['mots'],
        keyword8: json['keywords'][8]['mots'],
        keyword9: json['keywords'][9]['mots'],
        
       };
       this.toastr.success('The Context of your text is extracted Successfully !');/*success/error/warning/info/show()*/

       
     },
     error => {
       console.log(error);
       this.toastr.error('An error have occured in our servers please retry !');
       this.load=false;
     }
   );
   
  }
  public onChange(fileList: FileList): void {
    let file = fileList[0];
    let fileReader: FileReader = new FileReader();
    let self = this;
    fileReader.onloadend = function(x) {
      self.context.text = fileReader.result as string;
    }
    fileReader.readAsText(file);
  }

New(){
 
  this.router.navigate(['contextextraction']); 
}

}
